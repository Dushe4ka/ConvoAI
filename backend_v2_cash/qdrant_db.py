from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_ollama import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 30

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    timeout=QDRANT_TIMEOUT
)

collection_name = "knowledge_base"
embed_model = OllamaEmbeddings(model="llama3.2")

def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections.collections]

        if collection_name not in existing_collections:
            test_vector = embed_model.embed_query("test")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(test_vector),
                    distance=Distance.COSINE,
                    on_disk=True
                ),
                optimizers_config={
                    "default_segment_number": 2,
                    "indexing_threshold": 0
                }
            )
            logger.info(f"Коллекция '{collection_name}' создана")

            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="category",
                field_type="keyword"
            )
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="upload_date",
                field_type="datetime"
            )

        return True
    except Exception as e:
        logger.error(f"Ошибка инициализации Qdrant: {e}")
        raise RuntimeError(f"Не удалось подключиться к Qdrant: {e}")

def check_connection() -> bool:
    try:
        qdrant_client.get_collections()
        return True
    except Exception as e:
        logger.error(f"Ошибка подключения к Qdrant: {e}")
        return False










