# qdrant_db.py
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range, PayloadSchemaType
from langchain_ollama import OllamaEmbeddings
import logging
from tenacity import retry, stop_after_attempt, wait_fixed # Для повторных попыток подключения

# Импорт настроек из config.py
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_TIMEOUT,
    EMBEDDING_MODEL_NAME
)

logger = logging.getLogger(__name__)

# Глобальный клиент и модель эмбеддингов
qdrant_client: QdrantClient | None = None
embed_model: OllamaEmbeddings | None = None
VECTOR_SIZE: int | None = None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def initialize_qdrant_client() -> QdrantClient:
    """Инициализирует и возвращает клиент Qdrant."""
    global qdrant_client
    if qdrant_client is None:
        logger.info(f"Попытка подключения к Qdrant: host={QDRANT_HOST}, port={QDRANT_PORT}")
        try:
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                timeout=QDRANT_TIMEOUT,
                # Используйте https=True и api_key, если Qdrant настроен с безопасностью
                # https=False,
                # api_key=None,
            )
            client.get_collections() # Проверка соединения
            qdrant_client = client
            logger.info("Соединение с Qdrant успешно установлено.")
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            qdrant_client = None # Сбрасываем клиент при ошибке
            raise RuntimeError(f"Не удалось подключиться к Qdrant после нескольких попыток: {e}")
    return qdrant_client

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def initialize_embedding_model() -> OllamaEmbeddings:
    """Инициализирует и возвращает модель эмбеддингов."""
    global embed_model, VECTOR_SIZE
    if embed_model is None:
        logger.info(f"Инициализация модели эмбеддингов: {EMBEDDING_MODEL_NAME}")
        try:
            model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
            # Пробный запрос для определения размера и проверки работы
            test_vector = model.embed_query("test")
            VECTOR_SIZE = len(test_vector)
            embed_model = model
            logger.info(f"Модель эмбеддингов '{EMBEDDING_MODEL_NAME}' успешно инициализирована. Размер вектора: {VECTOR_SIZE}")
        except Exception as e:
            logger.error(f"Ошибка инициализации модели эмбеддингов '{EMBEDDING_MODEL_NAME}': {e}")
            embed_model = None
            VECTOR_SIZE = None
            raise RuntimeError(f"Не удалось инициализировать модель эмбеддингов: {e}")
    return embed_model

def get_qdrant_client() -> QdrantClient:
    """Возвращает инициализированный клиент Qdrant."""
    if qdrant_client is None:
        return initialize_qdrant_client()
    return qdrant_client

def get_embedding_model() -> OllamaEmbeddings:
    """Возвращает инициализированную модель эмбеддингов."""
    if embed_model is None:
        return initialize_embedding_model()
    return embed_model

def get_vector_size() -> int:
    """Возвращает размер вектора эмбеддингов."""
    if VECTOR_SIZE is None:
        initialize_embedding_model() # Убедимся, что модель инициализирована
    if VECTOR_SIZE is None: # Проверка после попытки инициализации
        raise ValueError("Не удалось определить размер вектора.")
    return VECTOR_SIZE

def create_qdrant_collection(collection_name: str):
    """Создает новую коллекцию в Qdrant, если она не существует."""
    client = get_qdrant_client()
    vec_size = get_vector_size() # Получаем размер вектора

    try:
        logger.info(f"Проверка/создание коллекции Qdrant: '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vec_size,
                distance=Distance.COSINE,
                on_disk=True # Рекомендуется для больших данных
            ),
            optimizers_config=models.OptimizersConfigDiff(
                 default_segment_number=2, # Оптимально для начала
                 indexing_threshold=20000 # Начать индексацию после N векторов (можно 0 для малых чатов)
            ),
            # replication_factor=1, # Для standalone инстанса
            # write_consistency_factor=1, # Для standalone инстанса
        )
        logger.info(f"Коллекция '{collection_name}' успешно создана.")

        # Создаем индексы для payload полей для быстрой фильтрации
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="category",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Индекс payload для 'category' в '{collection_name}' создан.")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="upload_date",
                field_schema=PayloadSchemaType.DATETIME
            )
            logger.info(f"Индекс payload для 'upload_date' в '{collection_name}' создан.")
        except Exception as index_e:
            # Если коллекция уже существовала с индексами, это нормально
            if "already exists" in str(index_e).lower() or "exists" in str(index_e).lower():
                 logger.warning(f"Индексы payload для '{collection_name}', возможно, уже существуют.")
            else:
                 logger.error(f"Ошибка создания индексов payload для '{collection_name}': {index_e}")
                 # Можно решить, критична ли эта ошибка

        return True

    except Exception as e:
        # Проверяем специфичную ошибку Qdrant об уже существующей коллекции
        if "already exists" in str(e).lower() or "Collection `"+collection_name+"` already exists!" in str(e):
             logger.warning(f"Коллекция '{collection_name}' уже существует.")
             # Убедимся, что индексы есть (на случай если коллекция была, а индексов нет)
             try:
                 client.create_payload_index(collection_name=collection_name, field_name="category", field_schema=PayloadSchemaType.KEYWORD)
                 client.create_payload_index(collection_name=collection_name, field_name="upload_date", field_schema=PayloadSchemaType.DATETIME)
             except Exception: pass # Игнорируем ошибки, если индексы уже есть
             return True # Считаем успехом, если уже есть
        logger.error(f"Непредвиденная ошибка при создании коллекции Qdrant '{collection_name}': {e}", exc_info=True)
        # Можно перевыбросить ошибку, если это критично для запуска
        # raise RuntimeError(f"Не удалось создать коллекцию Qdrant '{collection_name}': {e}")
        return False # Возвращаем False при других ошибках

def delete_qdrant_collection(collection_name: str) -> bool:
    """Удаляет коллекцию из Qdrant."""
    client = get_qdrant_client()
    logger.info(f"Попытка удаления коллекции Qdrant: '{collection_name}'")
    try:
        response = client.delete_collection(collection_name=collection_name, timeout=60) # Увеличим таймаут
        if response: # Успешное удаление возвращает True
            logger.info(f"Коллекция '{collection_name}' успешно удалена.")
            return True
        else:
            # Это может произойти, если коллекция не найдена, но API не выбросил исключение
            logger.warning(f"Удаление коллекции '{collection_name}' вернуло 'False', возможно, она уже была удалена.")
            return True # Считаем успехом, если ее нет
    except Exception as e:
        # Обрабатываем ошибку "не найдено" как успех
        if "not found" in str(e).lower() or "doesn't exist" in str(e).lower() or "Collection not found" in str(e):
            logger.warning(f"Коллекция '{collection_name}' не найдена для удаления (возможно, уже удалена).")
            return True
        logger.error(f"Ошибка удаления коллекции Qdrant '{collection_name}': {e}", exc_info=True)
        return False

def initialize_qdrant_check(general_collection_name: str):
    """
    Проверяет соединение с Qdrant и создает общую коллекцию, если ее нет.
    Вызывается при старте приложения.
    """
    logger.info("Инициализация соединения с Qdrant и проверка основной коллекции...")
    try:
        initialize_qdrant_client() # Устанавливаем соединение
        initialize_embedding_model() # Инициализируем модель и размер вектора
        # Проверяем/создаем основную коллекцию
        if not create_qdrant_collection(general_collection_name):
             # Если основную коллекцию не удалось создать, это критично
             raise RuntimeError(f"Не удалось создать/проверить основную коллекцию Qdrant: {general_collection_name}")
        logger.info(f"Qdrant инициализирован, основная коллекция '{general_collection_name}' проверена/создана.")
        return True
    except Exception as e:
        logger.error(f"Критическая ошибка инициализации Qdrant: {e}", exc_info=True)
        # Прерываем запуск приложения, если Qdrant недоступен
        raise RuntimeError(f"Сбой инициализации Qdrant: {e}")

# Импортируем PointStruct для использования в других модулях
__all__ = [
    "get_qdrant_client",
    "get_embedding_model",
    "create_qdrant_collection",
    "delete_qdrant_collection",
    "initialize_qdrant_check",
    "PointStruct",
    "Filter",
    "FieldCondition",
    "MatchValue",
    "Range"
]