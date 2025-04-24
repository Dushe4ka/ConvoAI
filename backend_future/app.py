# app.py
import io
from datetime import datetime

from PIL import Image
from fastapi import FastAPI, Request, Depends, HTTPException, Form, status, UploadFile
from fastapi.params import File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
import uuid
from typing import List, Optional, AsyncGenerator

from backend_v2 import chat_AI
# --- Импорт настроек и констант ---
from config import (
    GENERAL_CHAT_SESSION_ID, GENERAL_CHAT_NAME, GENERAL_QDRANT_COLLECTION,
    USER_CHAT_COLLECTION_PREFIX, MAX_IMAGE_SIZE, SUPPORTED_IMAGE_EXTENSIONS
)

# --- Импорт функций и моделей из других модулей ---
# База данных
from database import Base, engine, get_db, Chat, ChatMessage, create_database_tables, Document
from sqlalchemy.orm import Session
# Qdrant
from qdrant_db import initialize_qdrant_check, create_qdrant_collection, delete_qdrant_collection, get_qdrant_client, \
    get_embedding_model, PointStruct
# Чат AI
from chat_AI import chat_endpoint, get_chat_history, clear_chat_history, ai_memories, analyze_image_with_ollama
# Обработка файлов
from file_handler import upload_files_to_chat, get_categories
# Используем router из file_handler, если он там определен
# from .file_handler import router as file_router

# --- Настройка логгирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Инициализация FastAPI с lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Асинхронный менеджер контекста для событий startup и shutdown."""
    logger.info("Запуск приложения...")
    # === Startup ===
    try:
        # 1. Инициализация Qdrant (проверка соединения и основной коллекции)
        initialize_qdrant_check(GENERAL_QDRANT_COLLECTION)
        logger.info(f"Инициализация Qdrant завершена. Основная коллекция: '{GENERAL_QDRANT_COLLECTION}'")

        # 2. Создание таблиц в базе данных (если их нет)
        # В реальном продакшене используйте Alembic!
        create_database_tables()
        logger.info("Проверка/создание таблиц базы данных завершено.")

        # 3. Опционально: предзагрузка чего-либо еще

    except Exception as e:
        logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при запуске приложения: {e}", exc_info=True)
        # Можно решить, прерывать ли запуск или нет
        # raise RuntimeError(f"Сбой запуска: {e}") from e
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА ЗАПУСКА: {e} !!!\nПриложение может работать некорректно.\n")


    yield # Приложение работает здесь

    # === Shutdown ===
    logger.info("Остановка приложения...")
    # Здесь можно добавить код для очистки ресурсов при остановке
    # Например, закрытие соединений, сохранение состояния и т.д.


# Создание экземпляра FastAPI с lifespan
app = FastAPI(
    title="AI Knowledge Base",
    description="Система анализа документов с RAG-поиском и управлением чатами",
    version="1.0.0",
    lifespan=lifespan # Передаем менеджер контекста
)

# --- Подключение статических файлов и шаблонов ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Путь к фронтенду относительно папки с app.py
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend_future") # Убедитесь, что путь верный
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")

print(f"Static files directory: {STATIC_DIR}")
print(f"Directory exists: {os.path.exists(STATIC_DIR)}")
print(f"CSS file exists: {os.path.exists(os.path.join(STATIC_DIR, 'style.css'))}")

# Проверяем существование папок (опционально, но полезно для отладки)
if not os.path.isdir(FRONTEND_DIR): logger.warning(f"Папка фронтенда не найдена: {FRONTEND_DIR}")
if not os.path.isdir(STATIC_DIR): logger.warning(f"Папка static не найдена: {STATIC_DIR}")
if not os.path.isdir(TEMPLATES_DIR): logger.warning(f"Папка templates не найдена: {TEMPLATES_DIR}")

# Создаем папки, если их нет (только если уверены, что они должны быть здесь)
# os.makedirs(STATIC_DIR, exist_ok=True)
# os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Монтируем статику
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Инициализируем шаблонизатор
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Эндпоинты API ---

# === Управление Чатами ===

@app.get("/api/chats", summary="Получить список чатов", tags=["Chats"])
async def list_chats(db: Session = Depends(get_db)):
    """
    Возвращает список всех доступных чатов.
    Общий чат всегда первый в списке и помечен как is_general=True.
    Пользовательские чаты сортируются по дате создания (новые вверху).
    """
    try:
        # Запрашиваем пользовательские чаты из БД
        user_chats = db.query(Chat).order_by(Chat.created_at.desc()).all()

        # Формируем итоговый список для фронтенда
        chat_list = [
            {
                "session_id": GENERAL_CHAT_SESSION_ID,
                "name": GENERAL_CHAT_NAME,
                "is_general": True,
                "can_delete": False, # Общий чат нельзя удалить
                "created_at": None # Или дата старта приложения
            }
        ]
        # Добавляем пользовательские чаты
        chat_list.extend([
            {
                "session_id": chat.id,
                "name": chat.name,
                "is_general": False,
                "can_delete": True, # Пользовательские чаты можно удалять
                "created_at": chat.created_at.isoformat() if chat.created_at else None
            }
            # Исключаем случайное попадание ID общего чата в пользовательские (хотя его там быть не должно)
            for chat in user_chats if chat.id != GENERAL_CHAT_SESSION_ID
        ])
        return JSONResponse(content={"chats": chat_list})
    except Exception as e:
        logger.error(f"Ошибка получения списка чатов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось получить список чатов.")

@app.post("/api/chats", status_code=status.HTTP_201_CREATED, summary="Создать новый чат", tags=["Chats"])
async def create_new_chat_endpoint(
    name: Optional[str] = Form(None, description="Необязательное имя для нового чата."),
    db: Session = Depends(get_db)
):
    """
    Создает новый пользовательский чат:
    1. Генерирует уникальный ID (session_id).
    2. Создает соответствующую коллекцию в Qdrant.
    3. Сохраняет информацию о чате в базе данных.
    Возвращает информацию о созданном чате.
    """
    new_session_id = str(uuid.uuid4())
    chat_name = name.strip() if name and name.strip() else "Новый чат" # Имя по умолчанию или из формы
    # Генерируем уникальное имя для коллекции Qdrant, удаляя дефисы из UUID
    qdrant_collection = f"{USER_CHAT_COLLECTION_PREFIX}{new_session_id.replace('-', '')}"

    logger.info(f"Попытка создания нового чата: name='{chat_name}', session_id='{new_session_id}', qdrant_collection='{qdrant_collection}'")

    try:
        # 1. Создать коллекцию в Qdrant
        # Функция create_qdrant_collection вернет True, если успешно создана или уже существует
        if not create_qdrant_collection(qdrant_collection):
            # Если создание коллекции не удалось по причине, отличной от "already exists"
             raise HTTPException(status_code=500, detail=f"Не удалось создать инфраструктуру для чата (коллекция Qdrant '{qdrant_collection}').")

        # 2. Создать запись о чате в БД
        new_chat = Chat(
            id=new_session_id,
            name=chat_name,
            qdrant_collection_name=qdrant_collection
        )
        db.add(new_chat)
        db.commit() # Сохраняем изменения в БД
        db.refresh(new_chat) # Обновляем объект new_chat данными из БД (например, created_at)

        logger.info(f"Успешно создан новый чат: ID={new_chat.id}, Name='{new_chat.name}', Collection='{new_chat.qdrant_collection_name}'")
        # Возвращаем информацию о созданном чате
        return JSONResponse(content={
            "session_id": new_chat.id,
            "name": new_chat.name,
            "is_general": False,
            "can_delete": True,
            "qdrant_collection": new_chat.qdrant_collection_name, # Для отладки
            "created_at": new_chat.created_at.isoformat() if new_chat.created_at else None
        }, status_code=status.HTTP_201_CREATED)

    except HTTPException as http_exc:
         # Если ошибка пришла от create_qdrant_collection или другая HTTPException
         # Откат не требуется, т.к. коммита в БД еще не было или он вызвал ошибку
         db.rollback()
         logger.error(f"HTTP ошибка при создании чата '{chat_name}': {http_exc.detail}")
         raise http_exc
    except Exception as e:
        db.rollback() # Откатываем транзакцию БД
        logger.error(f"Критическая ошибка при создании чата '{chat_name}': {e}", exc_info=True)
        # Попытка отката: удалить коллекцию Qdrant, если она была создана до ошибки БД
        # Эта операция может не удаться, но стоит попытаться
        delete_qdrant_collection(qdrant_collection)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось создать чат из-за внутренней ошибки сервера.")


@app.delete("/api/chats/{session_id}", status_code=status.HTTP_200_OK, summary="Удалить пользовательский чат", tags=["Chats"])
async def delete_chat_endpoint(session_id: str, db: Session = Depends(get_db)):
    """
    Удаляет указанный пользовательский чат:
    1. Проверяет, что это не общий чат.
    2. Находит чат в БД.
    3. Удаляет чат и связанные с ним сообщения из БД (через cascade).
    4. Удаляет связанную коллекцию из Qdrant.
    5. Очищает память диалога для этого чата.
    """
    logger.info(f"Попытка удаления чата с ID: {session_id}")

    # 1. Запрет удаления общего чата
    if session_id == GENERAL_CHAT_SESSION_ID:
        logger.warning(f"Попытка удаления общего чата ({session_id}) отклонена.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Общий чат не может быть удален.")

    # 2. Поиск чата в БД
    chat_to_delete = db.query(Chat).filter(Chat.id == session_id).first()
    if not chat_to_delete:
        logger.warning(f"Попытка удаления несуществующего чата: {session_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Чат с ID {session_id} не найден.")

    # Сохраняем имя коллекции перед удалением объекта из сессии
    qdrant_collection_to_delete = chat_to_delete.qdrant_collection_name

    try:
        # 3. Удаление чата и сообщений из БД (cascade должен сработать)
        db.delete(chat_to_delete)
        db.commit() # Коммитим удаление из БД
        logger.info(f"Чат {session_id} удален из базы данных.")

        # 4. Очистка памяти диалога
        if session_id in ai_memories:
            ai_memories.pop(session_id, None)
            logger.info(f"Память для удаленного чата {session_id} очищена.")

        # 5. Удаление коллекции из Qdrant
        if qdrant_collection_to_delete:
             logger.info(f"Попытка удаления коллекции Qdrant '{qdrant_collection_to_delete}' для чата {session_id}")
             if delete_qdrant_collection(qdrant_collection_to_delete):
                 logger.info(f"Коллекция Qdrant '{qdrant_collection_to_delete}' успешно удалена или не найдена.")
             else:
                 # Если удаление коллекции не удалось, логируем ошибку, но не отменяем удаление чата
                 logger.error(f"Не удалось удалить коллекцию Qdrant '{qdrant_collection_to_delete}' для удаленного чата {session_id}. Возможно, потребуется ручная очистка.")
                 # Можно вернуть другой статус или сообщение об этом
        else:
             logger.warning(f"У удаленного чата {session_id} не было имени коллекции Qdrant для удаления.")


        return JSONResponse(content={"message": f"Чат '{session_id}' успешно удален."})

    except Exception as e:
        db.rollback() # Откатываем удаление из БД, если что-то пошло не так ПОСЛЕ поиска
        logger.error(f"Ошибка при удалении чата {session_id} или его ресурсов: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Не удалось полностью удалить чат из-за внутренней ошибки.")


# === Взаимодействие с чатом ===

@app.post("/api/chat", summary="Отправить сообщение в чат", tags=["Chat Interaction"])
async def handle_chat_message(request: chat_AI.ChatRequest, db: Session = Depends(get_db)):
     # Перенаправляем запрос в chat_endpoint из модуля chat_AI
     # Убедимся, что chat_AI.ChatRequest - это правильный Pydantic класс
     return await chat_endpoint(request=request, db=db)

@app.get("/api/history/{session_id}", summary="Получить историю сообщений чата", tags=["Chat Interaction"])
async def handle_get_history(session_id: str, db: Session = Depends(get_db)):
    # Перенаправляем запрос в get_chat_history из модуля chat_AI
    return await get_chat_history(session_id=session_id, db=db)

@app.delete("/api/chat/{session_id}", summary="Очистить историю сообщений чата", tags=["Chat Interaction"])
async def handle_clear_history(session_id: str, db: Session = Depends(get_db)):
    # Перенаправляем запрос в clear_chat_history из модуля chat_AI
    # Этот эндпоинт НЕ удаляет сам чат
    return await clear_chat_history(session_id=session_id, db=db)


# === Работа с файлами и категориями ===

# Используем router из file_handler, если он там есть и настроен
# app.include_router(file_router)

# Эндпоинт для ТЕКСТОВЫХ файлов (используем импортированную функцию)
@app.post("/api/upload", status_code=status.HTTP_207_MULTI_STATUS, summary="Загрузить текстовые файлы в чат", tags=["Files & Categories"])
async def handle_upload_files(
    session_id: str = Form(..., description="ID чата для загрузки"),
    files: List[UploadFile] = File(..., description="Текстовые файлы для загрузки (.txt, .pdf, .docx)"),
    db: Session = Depends(get_db)
):
     # Вызываем функцию из file_handler.py
     return await upload_files_to_chat(session_id=session_id, files=files, db=db)


# НОВЫЙ Эндпоинт для ИЗОБРАЖЕНИЙ
@app.post("/api/upload/image", status_code=status.HTTP_201_CREATED, summary="Загрузить и проанализировать изображение", tags=["Images"])
async def handle_upload_image(
    session_id: str = Form(..., description="ID чата (или 'general') для загрузки изображения"),
    image_file: UploadFile = File(..., description="Файл изображения (.png, .jpg, .jpeg, .webp)"),
    db: Session = Depends(get_db)
):
    """
    Загружает изображение, анализирует его с помощью Llama Vision,
    сохраняет описание и его эмбеддинг.
    """
    logger.info(f"Получен запрос на загрузку изображения в чат: {session_id}, файл: {image_file.filename}")

    # 1. Проверка типа файла
    file_extension = os.path.splitext(image_file.filename)[1].lower()
    if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.warning(f"Неподдерживаемый тип файла изображения: {image_file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый тип файла: {file_extension}. Поддерживаются: {', '.join(SUPPORTED_IMAGE_EXTENSIONS.keys())}"
        )

    # 2. Чтение и проверка размера
    image_bytes = await image_file.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        logger.warning(f"Файл изображения слишком большой: {image_file.filename} ({len(image_bytes)} байт)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл изображения слишком большой. Максимальный размер: {MAX_IMAGE_SIZE / 1024 / 1024:.2f} МБ."
        )
    await image_file.close() # Закрываем файл

    # 3. Базовая проверка, что это действительно изображение
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        logger.info(f"Файл '{image_file.filename}' успешно прошел базовую проверку изображения.")
        # Получаем реальный MIME тип после проверки Pillow
        actual_mime_type = Image.MIME.get(img.format)
        file_type = actual_mime_type or image_file.content_type # Используем реальный тип, если он определен
    except Exception as img_err:
        logger.error(f"Файл '{image_file.filename}' не является корректным изображением или поврежден: {img_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Файл '{image_file.filename}' не является корректным изображением или поврежден."
        )

    # 4. Определение коллекции Qdrant
    if session_id == GENERAL_CHAT_SESSION_ID or session_id.lower() == "general":
        collection_name = GENERAL_QDRANT_COLLECTION
        target_session_id = GENERAL_CHAT_SESSION_ID
    else:
        chat = db.query(Chat).filter(Chat.id == session_id).first()
        if not chat:
            logger.warning(f"Попытка загрузить изображение в несуществующий чат: {session_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Чат с ID {session_id} не найден.")
        collection_name = chat.qdrant_collection_name
        target_session_id = chat.id

    logger.info(f"Изображение будет добавлено в коллекцию Qdrant: '{collection_name}'")

    # 5. Анализ изображения с помощью Ollama Vision
    try:
        image_description = await analyze_image_with_ollama(image_bytes)
        if not image_description:
            logger.error(f"Не удалось получить описание для изображения '{image_file.filename}' от Ollama.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Не удалось проанализировать изображение с помощью ИИ."
            )
        logger.info(f"Получено описание для '{image_file.filename}'. Длина: {len(image_description)} символов.")
    except Exception as e:
        logger.error(f"Ошибка во время анализа изображения '{image_file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при обработке изображения: {e}"
        )

    # 6. Создание эмбеддинга для ОПИСАНИЯ
    try:
        embed_model = get_embedding_model()
        vector = embed_model.embed_query(image_description)
        logger.info(f"Создан эмбеддинг для описания изображения '{image_file.filename}'.")
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддинга для описания '{image_file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось создать векторное представление для описания изображения."
        )

    # 7. Сохранение информации в SQL базу данных
    db_document = None # Инициализируем
    try:
        upload_time = datetime.utcnow()
        db_document = Document(
            file_name=image_file.filename,
            file_type=file_type, # Используем тип, определенный Pillow
            upload_date=upload_time,
            is_image=True,
            image_description=image_description,
            category=None, # Категория не определяется для изображений
            chunks_count=1 # Описание считается одним "чанком"
            # chat_id=target_session_id # Если нужно
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        logger.info(f"Метаданные изображения '{image_file.filename}' сохранены в БД (ID: {db_document.id}).")
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при сохранении метаданных изображения '{image_file.filename}' в БД: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка сохранения информации об изображении в базе данных."
        )

    # 8. Сохранение в Qdrant (используем ID из БД)
    try:
        qdrant_client = get_qdrant_client()
        point_id = str(uuid.uuid4()) # Используем UUID для Qdrant ID, чтобы избежать коллизий если ID БД изменится
        # Или используем ID из БД, если уверены в его стабильности: point_id = str(db_document.id)

        # Формируем payload СОВМЕСТИМЫЙ с file_handler.py
        payload = {
            "text": image_description,           # Сгенерированное описание
            "source": image_file.filename,       # Имя файла
            "category": None,                    # Нет категории
            "upload_date": upload_time.isoformat(), # Время загрузки
            "chunk_index": 0,                    # Считаем одним чанком
            "is_image": True,                    # Флаг изображения
            "document_db_id": db_document.id,    # Ссылка на SQL запись
            "file_type": file_type               # Тип файла
        }

        point = PointStruct(id=point_id, vector=vector, payload=payload)

        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        logger.info(f"Эмбеддинг описания '{image_file.filename}' сохранен в Qdrant (ID: {point_id}) в коллекцию '{collection_name}'.")

    except Exception as e:
        logger.error(f"Ошибка при сохранении эмбеддинга '{image_file.filename}' в Qdrant: {e}", exc_info=True)
        # Важно: Нужно решить, удалять ли запись из БД в случае ошибки Qdrant
        # Простой вариант: оставляем запись в БД, но возвращаем ошибку
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка сохранения векторного представления в базе знаний."
        )

    # 9. Возвращаем успешный ответ
    return JSONResponse(
        status_code=status.HTTP_201_CREATED, # Используем 201 Created
        content={
            "message": "Изображение успешно загружено, проанализировано и проиндексировано.",
            "filename": image_file.filename,
            "description_preview": image_description[:250] + "..." if len(image_description) > 250 else image_description,
            "qdrant_collection": collection_name,
            "qdrant_point_id": point_id,
            "database_id": db_document.id
        })


# Эндпоинт для категорий (используем импортированную функцию)
@app.get("/api/categories", summary="Получить список категорий", tags=["Files & Categories"])
async def handle_get_categories():
    return get_categories()


# === Корневой эндпоинт для отображения HTML ===
@app.get("/", include_in_schema=False) # Не включаем в OpenAPI схему
async def read_root(request: Request):
    """Отдает главную HTML страницу."""
    try:
        return templates.TemplateResponse("index.html", {"request": request, "message": "Welcome"})
    except Exception as e:
         logger.error(f"Ошибка при рендеринге index.html: {e}", exc_info=True)
         # Отдаем простую ошибку, если шаблон не найден или не может быть обработан
         return JSONResponse(
             status_code=500,
             content={"detail": "Не удалось загрузить интерфейс приложения."}
         )

# --- Запуск приложения (если используется uvicorn напрямую) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск Uvicorn сервера для разработки...")
    # Определяем хост и порт (можно вынести в переменные окружения)
    APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    RELOAD_MODE = os.getenv("RELOAD_MODE", "true").lower() == "true"

    uvicorn.run(
        "app:app", # Путь к вашему FastAPI приложению
        host=APP_HOST,
        port=APP_PORT,
        reload=RELOAD_MODE, # Включить автоперезагрузку для разработки
        log_level="info" # Уровень логирования uvicorn
    )