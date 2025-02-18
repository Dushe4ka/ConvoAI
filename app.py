from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import uuid
import logging
import os
import fitz  # PyMuPDF
from docx import Document
from fastapi.responses import PlainTextResponse
import time
from functools import lru_cache

# --- Настройка логгера ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Инициализация FastAPI ---
app = FastAPI()

# --- Подключение статических файлов и шаблонов ---
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# --- Настройка БД SQLite с SQLAlchemy ---
DATABASE_URL = "sqlite:///./chat.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Модель чатов в БД ---
class ChatMessage(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # "user" или "ai"
    message = Column(Text)


# --- Создание таблицы чатов ---
Base.metadata.create_all(bind=engine)

# --- Qdrant (поиск векторов) ---
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
collection_name = "documents"
embed_model = OllamaEmbeddings(model="llama3.2")
llm = OllamaLLM(model="llama3.2")


# --- Инициализация коллекции в Qdrant ---
def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            test_vector = embed_model.embed_query("test")
            vector_size = len(test_vector)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),  # Указание косинусного расстояния
            )
            logger.info(f"Коллекция '{collection_name}' создана.")
        else:
            logger.info(f"Коллекция '{collection_name}' уже существует.")
    except Exception as e:
        logger.error(f"Ошибка инициализации Qdrant: {e}")


initialize_qdrant()


# --- Подключение БД к маршрутам ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Главная страница ---
@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Модель для сообщений ---
class ChatRequest(BaseModel):
    session_id: str
    message: str


# --- Фильтрация релевантных текстов ---
def filter_relevant_texts(search_results: list, max_length=512):
    relevant_texts = []
    for res in search_results:
        text = res.payload.get("text", "")
        if text and len(text.split()) < max_length:  # Ограничиваем длину фрагмента
            relevant_texts.append(text)
    return relevant_texts


# --- Генерация ответа с использованием RAG ---
async def generate_rag_response(user_message: str, search_results: list, embed_model, llm):
    relevant_texts = filter_relevant_texts(search_results)
    context = "\n".join(relevant_texts) + "\n\n" + user_message
    ai_response = await llm.invoke_async(context)  # Используем асинхронный метод для вызова LLM
    return ai_response


# --- Эндпоинт чата ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Пустой запрос")

    # --- Сохранение сообщения пользователя в БД ---
    db.add(ChatMessage(session_id=session_id, role="user", message=user_message))
    db.commit()

    try:
        # --- Векторный поиск в Qdrant ---
        query_embedding = embed_model.embed_query(user_message)
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5,
            score_threshold=0.7  # Порог схожести
        )

        if not search_results:
            # --- Генерация ответа от ИИ ---
            ai_response = llm.invoke(user_message)
        else:
            relevant_texts = ["Ты - преподаватель, используй следующие данные для ответа:"]
            for res in search_results:
                text = res.payload.get("text", "")
                relevant_texts.append(str(text))
            context = "\n".join(relevant_texts) + "\n\n" + user_message
            ai_response = llm.invoke(context)

        # --- Сохранение ответа ИИ в БД ---
        db.add(ChatMessage(session_id=session_id, role="ai", message=ai_response))
        db.commit()

        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        logger.error(f"Ошибка в обработке запроса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сервера")


# --- Загрузка файлов и их обработка ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # --- Читаем файл ---
        text = ""
        if file.filename.endswith(".pdf"):
            pdf_document = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in pdf_document])
            pdf_document.close()
        elif file.filename.endswith((".doc", ".docx")):
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # --- Разделение текста ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_text(text)

        # --- Сохранение в Qdrant ---
        for chunk in chunks:
            embedding = embed_model.embed_query(chunk)
            qdrant_client.upsert(collection_name=collection_name, points=[{
                "id": str(uuid.uuid4()), "vector": embedding, "payload": {"text": chunk}
            }])

        return JSONResponse(content={"message": "Файл успешно загружен и обработан."})

    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")


# --- Получение истории чатов ---
@app.get("/history/{session_id}")
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    return [{"role": msg.role, "message": msg.message} for msg in history]


# Очистка истории текущего чата
@app.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.commit()
    return PlainTextResponse("История чата очищена.")


@app.get("/new_chat")
async def create_new_chat():
    new_session_id = str(uuid.uuid4())  # Генерация уникального session_id
    collection_name = f"chat_{new_session_id}"  # Уникальная коллекция для чата

    try:
        # Проверяем, существует ли коллекция
        collections = qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            # Создаем новую коллекцию в Qdrant
            test_vector = embed_model.embed_query("test")  # Получаем размерность вектора
            vector_size = len(test_vector)

            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            logger.info(f"Создана коллекция: {collection_name}")

        return JSONResponse(content={"session_id": new_session_id, "collection": collection_name})

    except Exception as e:
        logger.error(f"Ошибка создания чата: {e}")
        raise HTTPException(status_code=500, detail="Ошибка создания чата")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
