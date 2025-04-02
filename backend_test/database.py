from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime, date
from passlib.context import CryptContext
from typing import Generator
import os

# Настройка подключения к БД
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./knowledge.db")

# Создание движка SQLAlchemy
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255))
    full_name = Column(String(100))
    birth_date = Column(Date)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    documents = relationship("Document", back_populates="owner")
    chat_messages = relationship("ChatMessage", back_populates="user")

    def verify_password(self, password: str):
        return pwd_context.verify(password, self.hashed_password)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255))
    stored_name = Column(String(255))
    file_type = Column(String(50))
    upload_date = Column(DateTime, default=datetime.utcnow)
    category = Column(String(100))
    subcategory = Column(String(100))
    vector_id = Column(String(36))
    chunks_count = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="documents")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), index=True)
    role = Column(String(10))
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column(Text)  # JSON с метаданными
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="chat_messages")

def get_db() -> Generator[Session, None, None]:
    """
    Генератор сессий для Dependency Injection в FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Инициализация базы данных (создание таблиц)
    """
    Base.metadata.create_all(bind=engine)

def create_first_admin():
    """
    Создание администратора по умолчанию
    """
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                hashed_password=pwd_context.hash("admin"),
                full_name="Admin",
                birth_date=date(2000, 1, 1),
                is_admin=True
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()