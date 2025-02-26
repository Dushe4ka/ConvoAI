from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import uuid
import logging
import os
import fitz  # PyMuPDF
from docx import Document

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

# --- Инициализация памяти для чатов с помощью langchain ---
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Инициализация локальной модели с помощью Ollama
llm = OllamaLLM(model="llama3.2", embedding_model="llama3.2")  # Укажите имя модели
conversation = ConversationChain(llm=llm, memory=memory)

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
        # --- Генерация ответа от ИИ с использованием памяти ---
        ai_response = conversation.predict(input=user_message)

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
    memory.clear()  # Очистка памяти для нового чата
    return JSONResponse(content={"session_id": new_session_id})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
