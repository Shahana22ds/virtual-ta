import glob
import logging
from openai import OpenAI
from qdrant_client.models import PointStruct
from .config import settings
from .qdrant import client, init_qdrant_collection
import pathlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key)

base_url = "https://tds.s-anand.net/#/"

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Chunk text into chunks of approximately 'size' characters,
    but only at sentence boundaries, with overlap between chunks.
    """
    import re

    sentences = re.findall(r'[^.!?]*[.!?]', text, re.DOTALL)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= size or not current_chunk:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            # Overlap: include last part of current_chunk in new chunk
            overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_text + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def ingest():
    # Recreate Qdrant collection
    init_qdrant_collection()
    
    # 1. Find all .txt files (adapt to .md, .html, etc.)
    files = glob.glob("data/raw/tds/**/*.txt", recursive=True)
    logging.info(f"Found {len(files)} files to ingest.")

    point_id = 0
    for path in files:
        logging.info(f"Processing file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        logging.info(f"Split into {len(chunks)} chunks.")

        batch_points = []

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

            # 4. Prepare points to batch and upsert
            batch_points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "source": url,
                        "text": chunk
                    }
                )
            )

            point_id += 1

            # Upsert in batches of 100 or at the last chunk
            if len(batch_points) == 100 or (i == len(chunks) and batch_points):
                try:
                    client.upsert(
                        collection_name="virtual_ta",
                        points=batch_points
                    )
                except Exception as e:
                    logging.error(f"Failed to upsert points batch ending at point {point_id-1} for file {path}: {e}")
                batch_points = []

    logging.info(f"✅ Ingested {point_id} chunks into Qdrant.")

if __name__ == "__main__":
    ingest()
