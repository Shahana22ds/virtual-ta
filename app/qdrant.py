from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from .config import settings
import logging

# Initialize client
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

def init_qdrant_collection():
    """(Re)create the 'virtual_ta' collection with cosine distance."""
    client.recreate_collection(
        collection_name="virtual_ta",
        vectors_config=VectorParams(
                    size=1536,             # embedding dimension
                    distance=Distance.COSINE,
                ),
    )
    logging.info("✅ Qdrant collection 'virtual_ta' is ready.")
    