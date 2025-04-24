# config.py
import uuid

# --- Константы для чатов ---
# Фиксированный ID для общего чата. Важно, чтобы он был постоянным.
GENERAL_CHAT_SESSION_ID: str = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
GENERAL_CHAT_NAME: str = "Общий чат"

# --- Константы Qdrant ---
# Имя коллекции в Qdrant для ОБЩЕГО чата
GENERAL_QDRANT_COLLECTION: str = "kb_general_chat_collection" # Выберите осмысленное имя
# Префикс для имен коллекций пользовательских чатов в Qdrant
USER_CHAT_COLLECTION_PREFIX: str = "kb_user_chat_"

# --- Настройки LLM и Embeddings ---
# Убедитесь, что используете одну и ту же модель везде
LLM_MODEL_NAME: str = "llama3.2" # Или ваша модель Ollama
EMBEDDING_MODEL_NAME: str = "llama3.2" # Или ваша модель Ollama

# --- Настройки Загрузки Файлов ---
MAX_FILE_SIZE_MB: int = 100 # Максимальный размер файла в МБ
MAX_FILE_SIZE: int = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS: dict = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain'
}
UPLOAD_DIR: str = "uploads" # Для временных файлов, если они понадобятся

# --- Настройки Qdrant Client ---
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
QDRANT_TIMEOUT: int = 30

# --- Настройки Базы Данных ---
DATABASE_URL: str = "sqlite:///./knowledge.db"

# --- Настройки Категорий ---
CATEGORIES: list[str] = [
    "Наука и технологии",
    "Бизнес и финансы",
    "Образование",
    "Здоровье",
    "Государство и право",
    "Творчество",
    "Разное"
]

# --- Настройки Чанков ---
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

QDRANT_BATCH_SIZE = 100  # Оптимальный размер батча
QDRANT_PAYLOAD_WARNING = 30 * 1024 * 1024  # 30MB - порог предупреждения

# Имя модели Llama Vision в Ollama (ЗАМЕНИТЕ НА ВАШЕ ТОЧНОЕ ИМЯ)
VISION_MODEL_NAME: str = "llama3.2-vision"
OLLAMA_BASE_URL: str = "http://localhost:11434"

MAX_IMAGE_SIZE_MB: int = 15
MAX_IMAGE_SIZE: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS: dict = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif'
}

IMAGE_ANALYSIS_PROMPT: str = (
    "Проанализируй это изображение максимально подробно. Опиши все видимые объекты, людей (если есть, их примерный возраст, одежду, действия), "
    "текст (если читаем), окружение (место, время суток, погода, если можно определить), "
    "возможные действия, события, общее настроение или атмосферу изображения. "
    "Структурируй описание. Будь объективным и исчерпывающим."
)