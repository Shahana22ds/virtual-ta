from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from app.config import settings
from app.qdrant import client
import re
from typing import Optional
import base64
import binascii
import logging
from app.utils import get_image_mime_type


app = FastAPI()
openai_client = OpenAI(api_key=settings.openai_api_key)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ---- Models ----
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class AnswerSourceLink(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: list[AnswerSourceLink]

# ---- Endpoint ----
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    logging.info("Received query request")
    # If image in query analyze the image and decode text
    image_resp_text = None
    image_mime_type = None
    if req.image is not None:
        image_mime_type = get_image_mime_type(req.image)
        logging.debug(f"Image MIME type detected: {image_mime_type}")
        try:
            base64.b64decode(req.image)
        except (binascii.Error, ValueError) as e:
            logging.error(f"Invalid base64 encoded image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoded image.\n{e}")
        prompt_messages = [
            {
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime_type};base64,{req.image}"
                    }
                }]
            },
            { "role": "user", "content": "what's in this image? disregard any non english text" }
        ]
        logging.info("Sending image content to OpenAI chat completion for image understanding")
        chat_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_messages,
            temperature=0.0,
        )
        image_resp_text = chat_resp.choices[0].message.content
        logging.debug(f"Image response text: {image_resp_text}")

    # 1. Embed the question
    input_text = req.question if image_resp_text is None else f"{image_resp_text} {req.question}"
    logging.info("Generating embedding for the question")
    emb_resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[input_text]
    )
    q_vector = emb_resp.data[0].embedding
    logging.debug(f"Embedding vector generated of length {len(q_vector)}")

    # 2. Retrieve top-5 similar chunks from Qdrant
    logging.info("Querying Qdrant for top-5 similar chunks")
    search_result = client.search(
        collection_name="virtual_ta",
        query_vector=q_vector,
        limit=5
    )

    if not search_result:
        logging.warning("No relevant articles found in Qdrant search")
        return QueryResponse(answer="No relevant articles found.", links=[])

    # 3. Build context passages with identifiers for source tracking
    contexts_with_ids = []
    for hit in search_result:
        payload = hit.payload or {}
        text_snippet = payload.get("text", "")
        # Use actual Qdrant point ID as the passage ID to avoid mismatch
        passage_id = hit.id if hasattr(hit, "id") else len(contexts_with_ids)
        contexts_with_ids.append({"id": passage_id, "text": text_snippet, "source": payload.get("source", "unknown")})
    logging.debug(f"Collected {len(contexts_with_ids)} context passages from search results")

    # 4. Prepare the prompt including markers for each passage to track relevant sources
    prompt_messages = []
    prompt_parts = []
    for item in contexts_with_ids:
        # Each passage prefixed with an ID marker, e.g., [Passage point_id]
        prompt_parts.append(f"[Passage {item['id']}]\n{item['text']}")
    system_prompt = (
        "You are a helpful TA. Use the following context passages to answer the question. "
        "If there is no relevant content, answer 'NO_DOCUMENTS_FOUND'. "
        "At the end of your answer, list only the IDs of the passages you actually used to answer, "
        "and list no others, in this format as the last line: 'SOURCES: [id1,id2,...]'\n\n"
        + "\n\n---\n\n".join(prompt_parts)
    )
    prompt_messages.append({"role": "system", "content": system_prompt})

    if req.image is not None:
        prompt_messages.extend([
        {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{req.image}"
                }
            }]
        },
        {
            "role": "user",
            "content": "Disregard any non english text in the image"
        }])
        logging.debug("Added image content to prompt messages")

    prompt_messages.append({"role": "user", "content": f"\n\nQuestion: {req.question}\nAnswer:"})

    logging.info("Sending prompt to OpenAI chat completion")
    chat_resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt_messages,
        temperature=0.0,
    )
    answer_full = chat_resp.choices[0].message.content
    if answer_full is None:
        logging.error("No answer returned from language model")
        raise HTTPException(status_code=500, detail="No answer returned from language model.")
    answer_full = answer_full.strip()
    logging.debug(f"Received answer from model: {answer_full}")

    if answer_full.startswith("NO_DOCUMENTS_FOUND"):
        logging.info("Model indicated no relevant articles found")
        return QueryResponse(answer="No relevant articles found.", links=[])

    # 5. Extract the SOURCES IDs from the model's answer
    source_match = re.search(r"SOURCES:\s*\[(.*?)\]", answer_full)
    if source_match:
        source_ids_str = source_match.group(1).strip()
        if source_ids_str:
            # Parse source IDs by their original IDs, which may not be sequential
            source_ids = []
            for s in source_ids_str.split(","):
                s_clean = s.strip()
                if s_clean.isdigit():
                    source_ids.append(int(s_clean))
                else:
                    # If IDs are non-integer strings, keep as string
                    source_ids.append(s_clean)
            logging.debug(f"Extracted source IDs from answer: {source_ids}")
        else:
            source_ids = []
            logging.debug("No source IDs found in the answer SOURCES field")
        # Remove the SOURCES part from the answer
        answer = re.sub(r"\n*SOURCES:\s*\[.*?\]", "", answer_full).strip()
    else:
        # If no sources provided by model, assume all
        source_ids = [item["id"] for item in contexts_with_ids]
        answer = answer_full
        logging.warning("No SOURCES field found in model answer, using all source IDs")

    # 6. Collect only the sources that were referenced
    # Match source_ids with contexts_with_ids by their actual 'id' field
    id_to_source = {item["id"]: AnswerSourceLink(url=item["source"], text=item["text"]) for item in contexts_with_ids}
    links = [id_to_source[sid] for sid in source_ids if sid in id_to_source]

    logging.info(f"Returning response with answer and {len(links)} source links")
    return QueryResponse(answer=answer, links=links)
