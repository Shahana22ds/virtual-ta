import glob
import logging
from openai import OpenAI
from qdrant_client.models import PointStruct
from .config import settings
from .qdrant import client, init_qdrant_collection
import pathlib
import json
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key)

base_url = "https://tds.s-anand.net/#/"

def chunk_text(text: str, size: int = 500, overlap: int = 300) -> list[str]:
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

def initialize_qdrant():
    """Recreate Qdrant collection."""
    init_qdrant_collection()
    logging.info("Qdrant collection initialized.")

def ingest_tds_data():
    """Ingest data from TDS files into Qdrant."""
    files = glob.glob("data/raw/tds/**/*.txt", recursive=True)
    logging.info(f"Found {len(files)} TDS files to ingest.")

    point_id = 0
    for path in files:
        logging.info(f"Processing TDS file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        logging.info(f"Split into {len(chunks)} chunks.")

        batch_points = []

        for i, chunk in enumerate(chunks, start=1):
            logging.info(f"Embedding chunk {i}/{len(chunks)} for file {path}")

            try:
                resp = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[chunk],
                    dimensions=1536,
                    encoding_format="float"
                )
                embedding = resp.data[0].embedding
            except Exception as e:
                logging.error(f"Failed to embed chunk {i} in file {path}: {e}")
                continue

            slug = '.'.join(pathlib.Path(path).name.replace("_", "/").split('.')[:-1])
            url = f"{base_url}{slug}"

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

            if len(batch_points) == 100 or (i == len(chunks) and batch_points):
                try:
                    client.upsert(
                        collection_name="virtual_ta",
                        points=batch_points
                    )
                except Exception as e:
                    logging.error(f"Failed to upsert points batch ending at point {point_id-1} for file {path}: {e}")
                batch_points = []

    logging.info(f"✅ Ingested {point_id} TDS chunks into Qdrant.")


def parse_replies_html_to_text(replies_html) -> str:
    """Convert a list of HTML replies to plain text separated by reply markers."""
    if not replies_html or not isinstance(replies_html, list):
        return ""

    plain_replies = []
    for reply_html in replies_html:
        soup = BeautifulSoup(reply_html, "html.parser")
        plain_replies.append(soup.get_text(separator="\n").strip())
    return "\n\n--- Reply ---\n\n".join(plain_replies)


def embed_chunk(chunk: str, post_index: int, chunk_index: int, total_chunks: int, path: str):
    """Create an embedding for a chunk of text, with logging and error handling."""
    logging.info(f"Embedding chunk {chunk_index}/{total_chunks} of post {post_index} from file {path}")
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[chunk],
            dimensions=1536,
            encoding_format="float"
        )
        return resp.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to embed chunk {chunk_index} of post {post_index} in file {path}: {e}")
        return None


def upsert_batch_points(batch_points, path, point_id):
    """Upsert batch points into Qdrant with error handling."""
    if not batch_points:
        return
    try:
        client.upsert(
            collection_name="virtual_ta",
            points=batch_points
        )
    except Exception as e:
        logging.error(f"Failed to upsert points batch ending at point {point_id - 1} for file {path}: {e}")


def process_post(post, post_index, total_posts, path, point_id, batch_points):
    """Process a single post: convert replies, chunk text, embed chunks, and prepare points."""
    raw_text = post.get("raw", "").strip()
    if not raw_text:
        return point_id

    replies_html = post.get("replies", [])
    replies_text = parse_replies_html_to_text(replies_html)

    combined_text = raw_text
    if replies_text:
        combined_text += "\n\n--- Replies ---\n\n" + replies_text

    chunks = chunk_text(combined_text)
    logging.info(f"Split post {post_index}/{total_posts} into {len(chunks)} chunks from file {path}")

    url = post.get("url") or base_url

    for j, chunk in enumerate(chunks, start=1):
        embedding = embed_chunk(chunk, post_index, j, len(chunks), path)
        if embedding is None:
            continue

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

        if len(batch_points) == 100:
            upsert_batch_points(batch_points, path, point_id)
            batch_points.clear()

    return point_id


def ingest_discourse_data():
    """Ingest Discourse data including raw text and replies converted from HTML to plain text."""
    files = glob.glob("data/raw/discourse/**/*.json", recursive=True)
    logging.info(f"Found {len(files)} Discourse JSON files to ingest.")

    point_id = 3000  # Start point_id to avoid conflict with tds point_ids

    for path in files:
        logging.info(f"Processing Discourse file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                posts = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load JSON from {path}: {e}")
                continue

        batch_points = []

        total_posts = len(posts)
        for i, post in enumerate(posts, start=1):
            point_id = process_post(post, i, total_posts, path, point_id, batch_points)

        # Upsert any remaining points after processing all posts in the file
        if batch_points:
            upsert_batch_points(batch_points, path, point_id)
            batch_points.clear()

    logging.info(f"✅ Ingested {point_id} Discourse posts into Qdrant.")

def ingest():
    initialize_qdrant()
    ingest_tds_data()
    ingest_discourse_data()

if __name__ == "__main__":
    ingest()
