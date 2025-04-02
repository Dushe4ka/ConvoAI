from fastapi import UploadFile, HTTPException, Depends, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import fitz  # PyMuPDF
from docx import Document
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional
from langchain_ollama import OllamaLLM
from pathlib import Path

from backend_test.auth_utils import get_current_user
from qdrant_db import qdrant_client, collection_name, embed_model
from database import get_db, Document as Doc, User
from chat_AI import DEFAULT_CATEGORIES

logger = logging.getLogger(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain'
}

# Инициализация модели для классификации
llm = OllamaLLM(model="llama3.2")


class FileUploadError(Exception):
    """Кастомное исключение для ошибок загрузки файлов"""
    pass


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Разбивает текст на перекрывающиеся чанки с сохранением целостности предложений"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    # Добавляем перекрытие между чанками
    if len(chunks) > 1 and overlap > 0:
        for i in range(1, len(chunks)):
            prev_sentences = chunks[i - 1].split('. ')
            overlap_sentences = prev_sentences[-overlap:] if len(prev_sentences) > overlap else prev_sentences
            chunks[i] = '. '.join(overlap_sentences) + '. ' + chunks[i]

    return chunks


async def process_file(file: UploadFile) -> Dict:
    """Обрабатывает загруженный файл и возвращает текст"""
    try:
        # Проверка расширения файла
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise FileUploadError(f"Неподдерживаемый формат файла: {file_ext}")

        # Создаем уникальное имя файла для избежания конфликтов
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Сохраняем файл временно
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if not content:
                raise FileUploadError("Файл пустой")
            buffer.write(content)

        # Извлекаем текст в зависимости от типа файла
        text = ""
        try:
            if file_ext == '.pdf':
                with fitz.open(file_path) as doc:
                    text = "\n".join(page.get_text() for page in doc)
            elif file_ext in ('.doc', '.docx', '.txt'):
                doc = Document(file_path)
                text = "\n".join(para.text for para in doc.paragraphs)
            else:  # .txt
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            if not text.strip():
                raise FileUploadError("Не удалось извлечь текст из файла")

            return {
                "text": text,
                "original_filename": file.filename,
                "saved_filename": unique_filename,
                "file_type": SUPPORTED_EXTENSIONS[file_ext],
                "file_size": len(content)
            }

        except Exception as e:
            raise FileUploadError(f"Ошибка обработки файла: {str(e)}")

    except Exception as e:
        logger.error(f"File processing error: {str(e)}", exc_info=True)
        raise
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)


async def upload_file(
        file: UploadFile = File(..., description="Файл для загрузки (PDF, DOCX, TXT)"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    try:
        # Обработка файла (оставляем без изменений)
        file_data = await process_file(file)
        text = file_data["text"]

        # Определение категорий (оставляем без изменений)
        main_category = detect_category(text)
        subcategory = detect_subcategory(text, main_category)

        # Разбиение на чанки (оставляем без изменений)
        chunks = chunk_text(text)
        if not chunks:
            raise FileUploadError("Не удалось разбить текст на чанки")

        # Подготовка данных для Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = embed_model.embed_query(chunk)
                point_id = str(uuid.uuid4())
                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk,
                        "source": file_data["original_filename"],
                        "category": main_category,
                        "subcategory": subcategory,
                        "date": datetime.now().isoformat(),
                        "chunk_index": i,
                        "uploaded_by": current_user.username  # Добавляем информацию о пользователе
                    }
                })
            except Exception as e:
                logger.error(f"Ошибка создания эмбеддинга для чанка {i}: {str(e)}")
                continue

        if not points:
            raise FileUploadError("Не удалось создать векторные представления для текста")

        # Сохранение в Qdrant
        try:
            operation_info = qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            if operation_info.status != 'completed':
                raise FileUploadError("Ошибка сохранения данных в векторной БД")
        except Exception as e:
            raise FileUploadError(f"Ошибка Qdrant: {str(e)}")

        # Сохранение метаданных в SQL с привязкой к пользователю
        try:
            doc = Doc(
                file_name=file_data["original_filename"],
                stored_name=file_data["saved_filename"],
                file_type=file_data["file_type"],
                upload_date=datetime.utcnow(),
                category=main_category,
                subcategory=subcategory,
                vector_id=str(uuid.uuid4()),
                chunks_count=len(points),
                user_id=current_user.id  # Добавляем user_id
            )
            db.add(doc)
            db.commit()
        except Exception as e:
            db.rollback()
            raise FileUploadError(f"Ошибка сохранения метаданных: {str(e)}")

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "filename": file_data["original_filename"],
                "details": {
                    "chunks": len(points),
                    "category": main_category,
                    "subcategory": subcategory,
                    "file_size": file_data["file_size"],
                    "file_type": file_data["file_type"],
                    "uploaded_by": current_user.username  # Добавляем информацию о пользователе
                }
            }
        )

    except FileUploadError as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при обработке файла"
        )

def get_categories():
    """Возвращает полную иерархию категорий"""
    return JSONResponse(content={"categories": DEFAULT_CATEGORIES})



def detect_category(text: str) -> str:
    """Определяет основную категорию текста с помощью LLM"""
    prompt = f"""
    Анализируй текст и определи наиболее подходящую категорию из списка.
    Отвечай ТОЛЬКО названием категории без пояснений.

    Доступные категории: {", ".join(DEFAULT_CATEGORIES.keys())}

    Текст для анализа: "{text[:1000]}"

    Категория:
    """
    try:
        category = llm.invoke(prompt).strip()
        return category if category in DEFAULT_CATEGORIES else "Разное"
    except Exception as e:
        logger.error(f"Ошибка определения категории: {e}")
        return "Разное"


def detect_subcategory(text: str, main_category: str) -> Optional[str]:
    """Определяет подкатегорию с помощью LLM"""
    if main_category not in DEFAULT_CATEGORIES:
        return None

    subcategories = list(DEFAULT_CATEGORIES[main_category].keys())
    if not subcategories:
        return None

    prompt = f"""
    Определи наиболее подходящую подкатегорию из списка для данного текста.
    Отвечай ТОЛЬКО названием подкатегории без пояснений.

    Основная категория: {main_category}
    Доступные подкатегории: {", ".join(subcategories)}

    Текст для анализа: "{text[:500]}"

    Подкатегория:
    """
    try:
        subcategory = llm.invoke(prompt).strip()
        return subcategory if subcategory in subcategories else None
    except Exception as e:
        logger.error(f"Ошибка определения подкатегории: {e}")
        return None