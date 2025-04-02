# database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base # Обновленный импорт
from datetime import datetime
import uuid

# Импорт настроек из config.py
from config import DATABASE_URL

# Используем declarative_base()
Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}, # Важно для SQLite
    pool_size=20, # Можно настроить
    max_overflow=30 # Можно настроить
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Модель Chat ---
class Chat(Base):
    __tablename__ = "chats"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String(100), default="Новый чат")
    qdrant_collection_name = Column(String(255), unique=True, nullable=False, index=True) # Добавлено nullable=False
    created_at = Column(DateTime, default=datetime.utcnow)

    # Связь с сообщениями для каскадного удаления
    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan", lazy="dynamic") # lazy="dynamic" для больших историй

# --- Модель ChatMessage ---
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    # Внешний ключ к Chat.id (UUID в виде строки)
    # Убедитесь, что этот столбец может хранить UUID как строку
    session_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String(10), nullable=False) # user или ai
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # metadata_ = Column(Text) # Убрано, т.к. не использовалось явно. Можно вернуть при необходимости.

    # Связь с чатом
    chat = relationship("Chat", back_populates="messages")

# --- Модель Document ---
# Хранит метаданные о загруженных файлах
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    # stored_name не нужен, так как файлы не хранятся постоянно
    # stored_name = Column(String(255))
    file_type = Column(String(100)) # Увеличен размер для MIME типов
    upload_date = Column(DateTime, default=datetime.utcnow)
    category = Column(String(100)) # Категория, определенная LLM
    # vector_id не нужен, т.к. у каждой точки в Qdrant свой ID
    # vector_id = Column(String(36))
    chunks_count = Column(Integer) # Количество чанков, на которые разбит файл
    # Опционально: связь с чатом, если нужно знать, в какой чат загружен файл
    # chat_id = Column(String(36), ForeignKey("chats.id"), nullable=True)

def get_db():
    """Зависимость FastAPI для получения сессии БД."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_database_tables():
    """Создает таблицы в базе данных, если их нет."""
    # В реальном проекте лучше использовать Alembic для миграций
    try:
        Base.metadata.create_all(bind=engine)
        print("Таблицы базы данных проверены/созданы.")
    except Exception as e:
        print(f"Ошибка при создании таблиц БД: {e}")
        raise

# Пример вызова для создания таблиц при первом запуске (или используйте Alembic)
# if __name__ == "__main__":
#     print("Создание таблиц базы данных...")
#     create_database_tables()
#     print("Готово.")