from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import logging
from langchain_ollama import OllamaEmbeddings

# --- Настройки ---
logger = logging.getLogger(__name__)
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
collection_name = "documents"
embed_model = OllamaEmbeddings(model="llama3.2")

# --- Инициализация Qdrant ---
def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            test_vector = embed_model.embed_query("test")
            qdrant_client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=len(test_vector), distance=Distance.COSINE))
            logger.info(f"Коллекция '{collection_name}' создана.")
    except Exception as e:
        logger.error(f"Ошибка Qdrant: {e}")
