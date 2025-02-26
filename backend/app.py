from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

from chat_AI import chat_endpoint, get_chat_history, clear_chat_history, create_new_chat
from file_handler import upload_file
from database import Base, engine
from qdrant_db import initialize_qdrant
import os

# --- Настройка логгера ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Инициализация FastAPI ---
app = FastAPI()

# --- Подключение статических файлов и шаблонов ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")  # Абсолютный путь к шаблонам
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Инициализация базы данных ---
Base.metadata.create_all(bind=engine)

# --- Инициализация Qdrant ---
initialize_qdrant()

# --- Подключение маршрутов ---
app.post("/chat")(chat_endpoint)
app.get("/history/{session_id}")(get_chat_history)
# app.delete("/chat/{session_id}")(clear_chat_history)
app.get("/new_chat")(create_new_chat)
app.post("/upload_file/")(upload_file)
app.delete("/clear_chat/{session_id}")(clear_chat_history)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
