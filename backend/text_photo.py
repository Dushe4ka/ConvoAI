from fastapi import UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import os
import fitz  # PyMuPDF
from docx import Document
import uuid
import logging

from qdrant_db import qdrant_client, collection_name
from chat_AI import embed_model, vision_model  # Подключаем LLaMA 3.2 и LLaMA 3.2-Vision

# --- Настройки ---
UPLOAD_DIR = "../uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Разбиение текста на чанки ---
def split_text(text, chunk_size=512, overlap=50):
    """Разбивает текст на перекрывающиеся чанки."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# --- Обработчик загрузки текстового файла ---
@router.post("/upload/file/")
async def upload_file(file: UploadFile = File(...)):
    """Загружает текстовый файл, извлекает текст и сохраняет в Qdrant."""
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

        # Разбиваем текст на чанки
        chunks = split_text(text)

        # Вставляем чанки в Qdrant
        points = []
        for chunk in chunks:
            embedding = embed_model.embed_query(chunk)  # Используем LLaMA 3.2
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"text": chunk}
            })

        qdrant_client.upsert(collection_name=collection_name, points=points)

        return JSONResponse(content={"message": f"Файл загружен, сохранено {len(chunks)} чанков."})

    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки файла")


# --- Обработчик загрузки изображения ---
@router.post("/upload/image/")
async def upload_image(file: UploadFile = File(...)):
    """Загружает изображение, получает описание через LLaMA 3.2-Vision и сохраняет в Qdrant."""
    file_ext = os.path.splitext(file.filename)[-1].lower()

    if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Неверный формат файла. Разрешены: JPG, PNG, WEBP")

    # Генерируем уникальное имя файла
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # Читаем порциями по 1MB
                buffer.write(chunk)

        # --- ОТПРАВКА В LLaMA 3.2-Vision ---
        prompt = (
            "Опиши это изображение максимально подробно. "
            "Перечисли все объекты, цвета, фон, действия людей и атмосферу сцены."
        )
        description = vision_model.describe_image(file_path, prompt)

        # Разбиваем описание на чанки
        chunks = split_text(description)

        # Вставляем чанки в Qdrant
        points = []
        for chunk in chunks:
            embedding = embed_model.embed_query(chunk)  # Создаём эмбеддинг текста
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"text": chunk, "image_path": file_path}
            })

        qdrant_client.upsert(collection_name=collection_name, points=points)

        logger.info(f"Изображение {file.filename} обработано. Описание сохранено в Qdrant.")

        return JSONResponse(content={"message": "Изображение загружено и обработано", "description": description})

    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки изображения")
