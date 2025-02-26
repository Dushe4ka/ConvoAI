from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import uuid
import logging
import os
import fitz  # PyMuPDF
from docx import Document
import time
import markdown  # новая библиотека для преобразования Markdown в HTML

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
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
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

# --- Глобальное хранилище памяти ИИ (по session_id) ---
ai_memories = {}

# --- Функция для конденсации (реформулирования) вопроса (из предыдущего примера) ---
def condense_question(user_message: str, memory: ConversationBufferMemory) -> str:
    if len(user_message.split()) < 5:
        history = memory.load_memory_variables({}).get("history", "")
        prompt = (f"На основе следующего контекста:\n{history}\n"
                  f"Сформулируй уточняющий и конкретный запрос для поиска дополнительных данных по теме. "
                  f"Исходный вопрос: '{user_message}'")
        refined_question = llm.invoke(prompt)
        return refined_question.strip()
    return user_message

# --- Функция для форматирования сообщения с Markdown ---
def format_message(message: str) -> str:
    """
    Преобразует текст, содержащий Markdown-разметку, в HTML.
    """
    return markdown.markdown(message)

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
        if text and len(text.split()) < max_length:
            relevant_texts.append(text)
    return relevant_texts

# --- Генерация ответа с использованием RAG и памяти ИИ ---
async def generate_rag_response(user_message: str, search_results: list, memory: ConversationBufferMemory):
    if search_results:
        retrieved_context = "Ты - преподаватель, используй следующие данные для ответа:\n" + "\n".join(
            [res.payload.get("text", "") for res in search_results]
        )
    else:
        retrieved_context = ""
    history = memory.load_memory_variables({}).get("history", "")
    combined_context = f"{history}\n{retrieved_context}\nUser: {user_message}"

    ai_response = llm.invoke(combined_context)

    if search_results:
        print("Контекст, использованный для формирования ответа:\n", combined_context)
        ai_response += "\n\n(ответ сформирован на основе данных из хранилища)"
    else:
        ai_response += "\n\n(Сгенерировала ИИ основываясь на своих знаниях)"

    # Преобразуем итоговый ответ с Markdown-разметкой в HTML
    formatted_response = format_message(ai_response)
    return formatted_response

# --- Эндпоинт чата ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Пустой запрос")

    db.add(ChatMessage(session_id=session_id, role="user", message=user_message))
    db.commit()

    if session_id not in ai_memories:
        ai_memories[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    memory = ai_memories[session_id]

    refined_query = condense_question(user_message, memory)
    if refined_query != user_message:
        logger.info(f"Вопрос преобразован из '{user_message}' в '{refined_query}' для поиска.")
    else:
        refined_query = user_message

    try:
        query_embedding = embed_model.embed_query(refined_query)
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5,
            score_threshold=0.7
        )

        ai_response = await generate_rag_response(user_message, search_results, memory)

        db.add(ChatMessage(session_id=session_id, role="ai", message=ai_response))
        db.commit()

        memory.save_context({"input": user_message}, {"output": ai_response})

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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_text(text)

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

# --- Очистка истории текущего чата (БД) ---
@app.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.commit()
    ai_memories.pop(session_id, None)
    return PlainTextResponse("История чата и память ИИ очищена.")

# --- Очистка памяти ИИ (истории контекста) без затрагивания БД ---
# @app.delete("/clear_memory/{session_id}")
# async def clear_ai_memory(session_id: str):
#     if session_id in ai_memories:
#         ai_memories[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
#     return PlainTextResponse("Память ИИ очищена.")

@app.get("/new_chat")
async def create_new_chat():
    new_session_id = str(uuid.uuid4())
    collection_name = f"chat_{new_session_id}"
    try:
        collections = qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            test_vector = embed_model.embed_query("test")
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
