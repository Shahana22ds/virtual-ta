from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from app.config import settings
from app.qdrant import client

app = FastAPI()
openai_client = OpenAI(api_key=settings.openai_api_key)

# ---- Models ----
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

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
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # 3. Build context & track sources
    contexts = []
    sources = []
    for hit in search_result:
        payload = hit.payload
        text_snippet = payload.get("text", "")
        source_path = payload.get("source", "unknown")
        contexts.append(text_snippet)
        sources.append(source_path)

    # 4. Call the chat API
    prompt = (
        "You are a helpful TA. Use the following context passages to answer the question. If there are no relevant content, let's answer 'NO_DOCUMENTS_FOUND'\n\n"
        + "\n\n---\n\n".join(contexts)
        + f"\n\nQuestion: {req.question}\nAnswer:"
    )
    chat_resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    answer = chat_resp.choices[0].message.content.strip()
    
    if answer == "NO_DOCUMENTS_FOUND":
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    return QueryResponse(answer=answer, sources=sources)
