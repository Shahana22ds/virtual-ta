import glob
import logging
from openai import OpenAI
from qdrant_client.models import PointStruct
from .config import settings
from .qdrant import client
import pathlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key)

base_url = "https://tds.s-anand.net/#/"

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
    logging.info(f"Found {len(files)} files to ingest.")

    point_id = 0
    for path in files:
        logging.info(f"Processing file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        logging.info(f"Split into {len(chunks)} chunks.")

        # 2. Chunk
        for i, chunk in enumerate(chunks, start=1):
            logging.info(f"Embedding chunk {i}/{len(chunks)} for file {path}")

            # 3. Embed
            try:
                resp = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[chunk]
                )
                embedding = resp.data[0].embedding  # extract vector  [oai_citation:0‡stackoverflow.com](https://stackoverflow.com/questions/77943395/openai-embeddings-api-how-to-extract-the-embedding-vector?utm_source=chatgpt.com)
            except Exception as e:
                logging.error(f"Failed to embed chunk {i} in file {path}: {e}")
                continue

            slug = '.'.join(pathlib.Path(path).name.replace("_", "/").split('.')[:-1])
            url = f"{base_url}{slug}"

            # 4. Prepare and upsert
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "source": url,
                    "text": chunk
                }
            )
            try:
                client.upsert(
                    collection_name="virtual_ta",
                    points=[point]                # batch of one; you can batch multiple
                )                               #  [oai_citation:1‡gist.github.com](https://gist.github.com/RGGH/cdbc24a873b4673bd137bdc1fb9bdd44?utm_source=chatgpt.com)
            except Exception as e:
                logging.error(f"Failed to upsert point {point_id} for file {path}: {e}")
                continue

            point_id += 1

    logging.info(f"✅ Ingested {point_id} chunks into Qdrant.")

if __name__ == "__main__":
    ingest()
