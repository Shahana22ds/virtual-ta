from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from app.config import settings
from app.qdrant import client
import re
from typing import Optional

app = FastAPI()
openai_client = OpenAI(api_key=settings.openai_api_key)

# ---- Models ----
class QueryRequest(BaseModel):
    question: str
    image: Optional[str]

class AnswerSourceLink(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: list[AnswerSourceLink]

# ---- Endpoint ----
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # 1. Embed the question
    emb_resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[req.question]
    )
    q_vector = emb_resp.data[0].embedding

    # 2. Retrieve top-5 similar chunks from Qdrant
    search_result = client.search(
        collection_name="virtual_ta",
        query_vector=q_vector,
        limit=5
    )

    if not search_result:
        return QueryResponse(answer="No relevant articles found.", links=[])

    # 3. Build context passages with identifiers for source tracking
    contexts_with_ids = []
    for hit in search_result:
        payload = hit.payload or {}
        text_snippet = payload.get("text", "")
        # Use actual Qdrant point ID as the passage ID to avoid mismatch
        passage_id = hit.id if hasattr(hit, "id") else len(contexts_with_ids)
        contexts_with_ids.append({"id": passage_id, "text": text_snippet, "source": payload.get("source", "unknown")})

    # 4. Prepare the prompt including markers for each passage to track relevant sources
    prompt_parts = []
    for item in contexts_with_ids:
        # Each passage prefixed with an ID marker, e.g., [Passage point_id]
        prompt_parts.append(f"[Passage {item['id']}]\n{item['text']}")
    prompt = (
        "You are a helpful TA. Use the following context passages to answer the question. "
        "If there is no relevant content, answer 'NO_DOCUMENTS_FOUND'. "
        "At the end of your answer, list only the IDs of the passages you actually used to answer, "
        "and list no others, in this format as the last line: 'SOURCES: [id1,id2,...]'\n\n"
        + "\n\n---\n\n".join(prompt_parts)
        + f"\n\nQuestion: {req.question}\nAnswer:"
    )

    print(prompt)

    chat_resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    answer_full = chat_resp.choices[0].message.content
    if answer_full is None:
        raise HTTPException(status_code=500, detail="No answer returned from language model.")
    answer_full = answer_full.strip()

    if answer_full.startswith("NO_DOCUMENTS_FOUND"):
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
        else:
            source_ids = []
        # Remove the SOURCES part from the answer
        answer = re.sub(r"\n*SOURCES:\s*\[.*?\]", "", answer_full).strip()
    else:
        # If no sources provided by model, assume all
        source_ids = [item["id"] for item in contexts_with_ids]
        answer = answer_full

    # 6. Collect only the sources that were referenced
    # Match source_ids with contexts_with_ids by their actual 'id' field
    id_to_source = {item["id"]: AnswerSourceLink(url=item["source"], text=item["text"]) for item in contexts_with_ids}
    links = [id_to_source[sid] for sid in source_ids if sid in id_to_source]

    return QueryResponse(answer=answer, links=links)
