import glob
from openai import OpenAI
from qdrant_client.models import PointStruct
from .config import settings
from .qdrant import client

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key)

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Naïve character-based chunker with overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def ingest():
    # 1. Find all .txt files (adapt to .md, .html, etc.)
    files = glob.glob("data/raw/**/*.txt", recursive=True)

    point_id = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # 2. Chunk
        for chunk in chunk_text(text):
            # 3. Embed
            resp = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[chunk]
            )
            embedding = resp.data[0].embedding  # extract vector  [oai_citation:0‡stackoverflow.com](https://stackoverflow.com/questions/77943395/openai-embeddings-api-how-to-extract-the-embedding-vector?utm_source=chatgpt.com)

            # 4. Prepare and upsert
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "source": path,
                    "text": chunk
                }
            )
            client.upsert(
                collection_name="virtual_ta",
                points=[point]                # batch of one; you can batch multiple
            )                               #  [oai_citation:1‡gist.github.com](https://gist.github.com/RGGH/cdbc24a873b4673bd137bdc1fb9bdd44?utm_source=chatgpt.com)

            point_id += 1

    print(f"✅ Ingested {point_id} chunks into Qdrant.")

if __name__ == "__main__":
    ingest()
