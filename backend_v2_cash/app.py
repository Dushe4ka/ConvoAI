from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os

from file_handler import upload_file, get_categories
from chat_AI import chat_endpoint, get_chat_history, clear_chat_history, create_new_chat
from database import Base, engine
from qdrant_db import initialize_qdrant

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Knowledge Base",
             description="Система анализа документов с RAG-поиском")

# Подключение статических файлов
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend_v2", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "frontend_v2", "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Инициализация БД
Base.metadata.create_all(bind=engine)
initialize_qdrant()

# Регистрация маршрутов
app.post("/api/chat")(chat_endpoint)
app.get("/api/history/{session_id}")(get_chat_history)
app.delete("/api/chat/{session_id}")(clear_chat_history)
app.post("/api/upload")(upload_file)
app.get("/api/categories")(get_categories)
app.get("/api/new_chat")(create_new_chat)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def lifespan():
    logger.info("Starting up...")
    try:
        initialize_qdrant()
        logger.info("Qdrant connection established")
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
