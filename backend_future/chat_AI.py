# chat_AI.py
import base64

import httpx
from fastapi import HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pydantic import BaseModel
import logging
import uuid
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dateutil.parser import parse

# --- Импорты из проекта ---
from database import get_db, ChatMessage, Chat
from qdrant_db import ( # Используем функции-геттеры
    get_qdrant_client, get_embedding_model,
    PointStruct, Filter, FieldCondition, MatchValue, Range
)
# --- Импорт настроек и констант ---
from config import (
    GENERAL_CHAT_SESSION_ID, GENERAL_QDRANT_COLLECTION, USER_CHAT_COLLECTION_PREFIX,
    CATEGORIES, LLM_MODEL_NAME, OLLAMA_BASE_URL, IMAGE_ANALYSIS_PROMPT, VISION_MODEL_NAME
)

logger = logging.getLogger(__name__)

# --- Инициализация LLM ---
# LLM инициализируется один раз при загрузке модуля
try:
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    logger.info(f"LLM модель '{LLM_MODEL_NAME}' успешно инициализирована.")
except Exception as e:
    logger.error(f"Критическая ошибка инициализации LLM модели '{LLM_MODEL_NAME}': {e}")
    # Можно прервать запуск или использовать заглушку
    llm = None # Установить в None, чтобы проверить перед использованием

# Словарь для хранения памяти каждого чата
ai_memories: Dict[str, ConversationBufferMemory] = {}

# --- Модели Pydantic ---
class ChatRequest(BaseModel):
    session_id: str # ID чата (Общего или пользовательского)
    message: str
    filters: Optional[dict] = None

class SearchFilters(BaseModel):
    category: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

# --- Вспомогательные функции ---

def get_qdrant_collection_for_chat(session_id: str, db: Session) -> str:
    """Определяет имя коллекции Qdrant для данного session_id."""
    if session_id == GENERAL_CHAT_SESSION_ID:
        # Для общего чата используем предопределенное имя коллекции
        return GENERAL_QDRANT_COLLECTION
    else:
        # Для пользовательского чата ищем запись в БД
        chat = db.query(Chat).filter(Chat.id == session_id).first()
        if not chat:
            logger.warning(f"Запрос к несуществующему чату: {session_id}")
            raise HTTPException(status_code=404, detail=f"Чат с ID {session_id} не найден")
        if not chat.qdrant_collection_name:
             # Эта ситуация не должна возникать при правильной работе API создания чата
             logger.error(f"Критическая ошибка: Чат {session_id} существует, но не имеет имени коллекции Qdrant.")
             raise HTTPException(status_code=500, detail=f"Ошибка конфигурации чата {session_id}: отсутствует имя коллекции Qdrant.")
        return chat.qdrant_collection_name

def extract_search_params(text: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Извлекает категорию и даты из текста запроса с помощью LLM."""
    if not llm:
        logger.error("LLM не инициализирована, не могу извлечь параметры поиска.")
        return None, None

    prompt = f"""
    Проанализируй следующий текст запроса пользователя.
    1. Определи наиболее подходящую категорию из списка: {CATEGORIES}. Если ни одна не подходит, выбери 'Разное'.
    2. Если в тексте явно указан период времени (например, "за прошлый год", "в марте 2023", "с 10 по 15 января"), определи даты начала и конца периода в формате YYYY-MM-DD. Если указана только одна дата, используй ее как начало или конец в зависимости от контекста. Если период не указан, оставь даты пустыми.

    Верни ответ СТРОГО в формате JSON со следующими ключами: "category" (строка), "dates" (объект с ключами "start" и "end", оба строки или null).

    Текст запроса (используй только первые 500 символов): "{text[:500]}"

    JSON ответ:
    """
    try:
        response = llm.invoke(prompt)
        # Попытка распарсить JSON из ответа LLM
        import json
        try:
             result = json.loads(response.strip())
        except json.JSONDecodeError:
             logger.warning(f"LLM вернула невалидный JSON для извлечения параметров: {response}")
             # Попытка извлечь через eval как запасной вариант (менее безопасный)
             try:
                 result = eval(response.strip())
                 if not isinstance(result, dict): result = {}
             except Exception:
                 result = {}

        category = result.get("category") if result.get("category") in CATEGORIES else None

        dates_extracted = result.get("dates")
        dates = None
        if isinstance(dates_extracted, dict):
            start_date_str = dates_extracted.get("start")
            end_date_str = dates_extracted.get("end")
            start_date, end_date = None, None
            try:
                if start_date_str: start_date = parse(start_date_str).strftime('%Y-%m-%d')
            except Exception: logger.warning(f"Не удалось распарсить дату начала: {start_date_str}")
            try:
                 if end_date_str: end_date = parse(end_date_str).strftime('%Y-%m-%d')
            except Exception: logger.warning(f"Не удалось распарсить дату конца: {end_date_str}")

            if start_date or end_date:
                 dates = {"start": start_date, "end": end_date}

        logger.info(f"Извлеченные параметры: Категория={category}, Даты={dates}")
        return category, dates

    except Exception as e:
        logger.error(f"Ошибка LLM при анализе запроса на параметры: {e}", exc_info=True)
        return None, None

async def search_in_qdrant(
        collection_name: str,
        query: str,
        category: Optional[str] = None,
        upload_dates: Optional[Dict[str, Optional[str]]] = None,
        limit: int = 15 # Можно увеличить, если результаты поиска слабые
) -> List[Dict]:
    """Поиск документов с фильтрацией в указанной коллекции Qdrant."""
    qdrant = get_qdrant_client()
    embed = get_embedding_model()
    must_conditions = []

    if category and category != "Разное":
        must_conditions.append(FieldCondition(
            key="category", # Фильтр по категории
            match=MatchValue(value=category)
        ))

    if upload_dates:
        date_filter_conditions = {}
        if upload_dates.get("start"):
            date_filter_conditions["gte"] = upload_dates["start"] # Формат YYYY-MM-DD
        if upload_dates.get("end"):
            date_filter_conditions["lte"] = upload_dates["end"] # Формат YYYY-MM-DD

        if date_filter_conditions:
            must_conditions.append(FieldCondition(
                # Убедитесь, что поле называется 'upload_date' в payload Qdrant
                key="upload_date",
                range=Range(**date_filter_conditions)
            ))

    try:
        query_embedding = embed.embed_query(query)
        logger.debug(f"Поиск в Qdrant: collection='{collection_name}', filters={must_conditions}, limit={limit}")

        search_results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True
        )

        # Адаптируем извлечение данных, чтобы включить is_image
        rag_context = []
        for hit in search_results:
             payload = hit.payload or {} # Безопасное извлечение payload
             rag_context.append({
                 "text": payload.get("text", ""),
                 # Поля извлекаются напрямую из payload (как в file_handler.py)
                 "source": payload.get("source", "unknown"),
                 "upload_date": payload.get("upload_date"),
                 "category": payload.get("category"), # Будет None для изображений
                 "is_image": payload.get("is_image", False), # <--- Добавляем флаг
                 "score": hit.score
             })

        logger.info(f"Найдено {len(rag_context)} релевантных фрагментов в '{collection_name}'.")
        return rag_context

    except Exception as e:
        if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
             logger.warning(f"Коллекция '{collection_name}' не найдена во время поиска.")
             return []
        logger.error(f"Ошибка поиска в Qdrant (коллекция: {collection_name}): {e}", exc_info=True)
        raise e


async def generate_rag_response(
        query: str,
        context: List[Dict],
        chat_history: str,
        category: Optional[str] = None,
        upload_dates: Optional[Dict[str, str]] = None
) -> str:
    """Генерирует RAG-ответ с учетом типа источника (текст/изображение)."""
    if not llm:
         logger.error("LLM не инициализирована, не могу сгенерировать ответ.")
         return "Извините, произошла ошибка при обработке вашего запроса (LLM недоступна)."

    # Адаптируем форматирование контекста
    context_lines = []
    for idx, item in enumerate(context):
        source_type = "Изображение" if item.get('is_image') else "Документ" # <--- Учитываем флаг
        source_name = item.get('source', 'неизвестно')
        category_info = f"Категория: {item.get('category', 'не указана')}" if item.get('category') else "Категория: -"
        date_info = f"Дата: {item.get('upload_date', 'неизвестно')[:10] if item.get('upload_date') else 'неизвестно'}"
        score_info = f"Релевантность: {item.get('score', 0.0):.2f}"

        header = f"Источник {idx+1} ({source_type}): {source_name} ({category_info}, {date_info}, {score_info})"
        # Для изображений "текст" - это их описание
        fragment_label = "Описание" if item.get('is_image') else "Фрагмент"
        fragment = f"{fragment_label}: {item.get('text', '')[:500]}..." # Ограничиваем длину

        context_lines.append(f"{header}\n{fragment}")

    context_str = "\n\n".join(context_lines) if context else "В базе знаний не найдено релевантной информации."

    filters_info = []
    if category and category != "Разное":
        filters_info.append(f"Применен фильтр по категории: {category}")
    if upload_dates:
        start = upload_dates.get('start', 'любой')
        end = upload_dates.get('end', 'любой')
        filters_info.append(f"Применен фильтр по дате загрузки: с {start} по {end}")
    filters_str = "\n".join(filters_info) if filters_info else "Фильтры не применялись."

    prompt = f"""
Ты - ИИ-ассистент базы знаний. Твоя задача - ответить на вопрос пользователя, основываясь на предоставленной истории диалога и найденных фрагментах документов или описаниях изображений.

ИСТОРИЯ ДИАЛОГА (последние сообщения):
{chat_history if chat_history else "Новый диалог."}

НАЙДЕННЫЕ ФРАГМЕНТЫ И ОПИСАНИЯ ИЗОБРАЖЕНИЙ:
{filters_str}
---
{context_str}
---

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
"{query}"

ИНСТРУКЦИИ ПО ФОРМИРОВАНИЮ ОТВЕТА:
1. Внимательно изучи историю диалога и найденные данные (фрагменты текста, описания изображений).
2. Отвечай ТОЛЬКО на основе предоставленной информации. Не придумывай факты.
3. Если найденные данные содержат ответ на вопрос, сформулируй его четко и лаконично.
4. Если информация релевантна, но не отвечает на вопрос прямо, укажи это.
5. Если релевантной информации нет или ее недостаточно, сообщи пользователю об этом прямо.
6. При использовании информации из источников, ССЫЛАЙСЯ на них в формате [Источник N], например: "На изображении [Источник 1] видно...", "В документе [Источник 2] сказано...".
7. Старайся поддерживать контекст диалога.
8. Ответ должен быть на русском языке.

ТВОЙ ОТВЕТ:
"""
    try:
        logger.debug(f"Генерация RAG-ответа. Промпт передан в LLM (длина: {len(prompt)}).")
        response = llm.invoke(prompt).strip()
        logger.info(f"LLM сгенерировала ответ (длина: {len(response)}).")
        return response
    except Exception as e:
        logger.error(f"Ошибка LLM при генерации RAG-ответа: {e}", exc_info=True)
        return "Извините, произошла ошибка при генерации ответа."


def generate_search_keywords(question: str, category: Optional[str] = None) -> str:
    """Генерирует ключевые слова для поиска в Qdrant с помощью LLM."""
    if not llm:
        logger.error("LLM не инициализирована, не могу сгенерировать ключевые слова.")
        return question # Возвращаем исходный вопрос как запасной вариант

    prompt = f"""
Извлеки самые важные ключевые слова и фразы из вопроса пользователя, которые лучше всего подойдут для поиска релевантных документов в векторной базе данных.
Учитывай возможную категорию документов: {category if category else "Любая"}.
Сфокусируйся на существительных, глаголах, терминах, названиях. Убери общие слова и вопросительные конструкции.
Ответ дай в виде строки, где ключевые слова/фразы разделены пробелом. Используй русский язык.

Исходный вопрос:
"{question}"

Ключевые слова для поиска:
"""
    try:
        keywords = llm.invoke(prompt).strip().replace(",", "") # Убираем запятые на всякий случай
        logger.info(f"Сгенерированы ключевые слова для поиска: '{keywords}'")
        return keywords if keywords else question # Если LLM вернула пустоту, используем исходный вопрос
    except Exception as e:
        logger.error(f"Ошибка LLM при генерации ключевых слов: {e}")
        return question # Возвращаем исходный вопрос при ошибке

def condense_question(user_message: str, memory: ConversationBufferMemory) -> str:
    """Уточняет и переформулирует вопрос пользователя с учетом истории диалога (если необходимо)."""
    if not llm:
        logger.error("LLM не инициализирована, не могу уточнить вопрос.")
        return user_message

    # Получаем историю из памяти
    history_messages = memory.chat_memory.messages
    history_str = "\n".join([f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in history_messages[-4:]]) # Берем последние 4 сообщения

    if not history_str: # Если истории нет, не нужно переформулировать
        return user_message

    prompt = f"""
Прочитай историю диалога и последний вопрос пользователя.
Если последний вопрос САМОДОСТАТОЧЕН и понятен без контекста истории, верни его БЕЗ ИЗМЕНЕНИЙ.
Если последний вопрос НЕПОНЯТЕН без контекста (например, использует местоимения "он", "она", "это" или ссылается на предыдущие ответы), переформулируй его в самостоятельный, полный вопрос, объединив с нужной информацией из истории.
Ответ должен быть только переформулированным вопросом. Не добавляй никаких пояснений.

ИСТОРИЯ ДИАЛОГА (последние сообщения):
{history_str}

ПОСЛЕДНИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:
"{user_message}"

УТОЧНЕННЫЙ (или исходный) ВОПРОС:
"""
    try:
        condensed_q = llm.invoke(prompt).strip()
        if condensed_q != user_message:
            logger.info(f"Вопрос пользователя уточнен: '{condensed_q}' (Исходный: '{user_message}')")
        return condensed_q
    except Exception as e:
        logger.error(f"Ошибка LLM при уточнении вопроса: {e}")
        return user_message # Возвращаем исходный вопрос при ошибке

# --- Основные эндпоинты API ---

async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """Обрабатывает сообщение пользователя, выполняет RAG и возвращает ответ AI."""
    session_id = request.session_id
    user_message = request.message.strip()

    if not llm: # Проверка инициализации LLM
        raise HTTPException(status_code=503, detail="Сервис временно недоступен (LLM не инициализирована).")

    if not user_message or len(user_message) < 3:
        raise HTTPException(status_code=400, detail="Запрос слишком короткий. Пожалуйста, сформулируйте вопрос подробнее.")

    # 1. Определяем коллекцию Qdrant для чата
    try:
        target_collection = get_qdrant_collection_for_chat(session_id, db)
    except HTTPException as e:
         raise e # Пробрасываем 404 или 500, если чат не найден или некорректен
    except Exception as e:
        logger.error(f"Не удалось определить коллекцию для чата {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при определении чата.")

    # 2. Сохраняем сообщение пользователя в БД
    try:
        db_message = ChatMessage(session_id=session_id, role="user", message=user_message)
        db.add(db_message)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка сохранения сообщения пользователя в БД для чата {session_id}: {e}", exc_info=True)
        # Не прерываем выполнение, но логируем
        # Можно вернуть ошибку, если сохранение критично

    # 3. Управление памятью диалога
    if session_id not in ai_memories:
        logger.info(f"Инициализация памяти для чата {session_id}")
        ai_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Опционально: Загрузить историю из БД в память при первой инициализации
        # history_db = await get_chat_history(session_id, db) # Используем существующую функцию
        # for msg in history_db:
        #     if msg['role'] == 'user': ai_memories[session_id].chat_memory.add_user_message(msg['message'])
        #     elif msg['role'] == 'ai': ai_memories[session_id].chat_memory.add_ai_message(msg['message'])
    memory = ai_memories[session_id]

    try:
        # 4. Обработка фильтров из запроса (если есть)
        filters = request.filters or {}
        category_filter = filters.get("category")
        upload_dates_filter = None
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if date_from or date_to:
            upload_dates_filter = {"start": date_from, "end": date_to}

        # 5. Уточнение вопроса с учетом истории
        condensed_user_message = condense_question(user_message, memory)

        # 6. Автоматическое определение параметров поиска (если фильтры не заданы)
        auto_category, auto_dates = None, None
        if not category_filter and not upload_dates_filter: # Определяем параметры, только если фильтры не заданы вручную
             auto_category, auto_dates = extract_search_params(condensed_user_message)

        # Приоритет ручным фильтрам
        final_category = category_filter if category_filter else auto_category
        final_dates = upload_dates_filter if upload_dates_filter else auto_dates

        # 7. Генерация ключевых слов для поиска
        search_query = generate_search_keywords(condensed_user_message, final_category)

        logger.info(f"Чат ID: {session_id} | Коллекция: {target_collection}")
        logger.info(f"Обработка запроса: '{user_message}' (Уточненный: '{condensed_user_message}')")
        logger.info(f"Параметры поиска: Query='{search_query}', Category='{final_category}', Dates='{final_dates}'")

        # 8. Поиск релевантных документов в Qdrant
        search_results = await search_in_qdrant(
            collection_name=target_collection,
            query=search_query,
            category=final_category,
            upload_dates=final_dates,
            limit=10 # Уменьшим лимит для RAG
        )

        # 9. Генерация ответа AI с использованием RAG
        history_for_rag = memory.load_memory_variables({}).get("chat_history", "")
        history_str = "\n".join([f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in history_for_rag[-6:]]) # Последние 6 сообщений для контекста LLM

        ai_response = await generate_rag_response(
            query=user_message, # Отвечаем на исходный вопрос пользователя
            context=search_results,
            chat_history=history_str,
            category=final_category,
            upload_dates=final_dates
        )

        # 10. Сохранение ответа AI в БД и памяти
        try:
            db_ai_message = ChatMessage(session_id=session_id, role="ai", message=ai_response)
            db.add(db_ai_message)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Ошибка сохранения сообщения AI в БД для чата {session_id}: {e}", exc_info=True)
            # Не прерываем, но логируем

        memory.save_context({"input": user_message}, {"output": ai_response})

        # 11. Формирование ответа API
        metadata = {
            "search_query": search_query,
            "applied_category": final_category,
            "applied_dates": final_dates,
            "retrieved_sources": list(set(res['source'] for res in search_results)),
            "num_retrieved": len(search_results),
            "qdrant_collection": target_collection
        }

        return JSONResponse(content={
            "response": ai_response,
            "metadata": metadata
        })

    except HTTPException as e:
         # Пробрасываем HTTP исключения (например, 404 от get_qdrant_collection_for_chat)
         raise e
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке запроса в чате {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке вашего запроса.")


async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Получает историю сообщений для указанного чата, отсортированную по времени."""
    # Проверка существования чата (опционально, но улучшает UX)
    if session_id != GENERAL_CHAT_SESSION_ID:
         chat = db.query(Chat).filter(Chat.id == session_id).first()
         if not chat:
             raise HTTPException(status_code=404, detail=f"Чат с ID {session_id} не найден")

    try:
        # Запрос сообщений с сортировкой по дате создания
        history = db.query(ChatMessage)\
                    .filter(ChatMessage.session_id == session_id)\
                    .order_by(ChatMessage.created_at.asc())\
                    .all()

        # Формирование ответа
        return [
            {
                "role": msg.role,
                "message": msg.message,
                "created_at": msg.created_at.isoformat() # Добавляем время для отображения
             }
            for msg in history
        ]
    except Exception as e:
        logger.error(f"Ошибка получения истории чата {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось получить историю чата.")


async def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Очищает ТОЛЬКО сообщения указанного чата, не удаляя сам чат или коллекцию Qdrant."""
    logger.info(f"Попытка очистки истории для чата {session_id}")

    # Проверка существования чата (опционально)
    if session_id != GENERAL_CHAT_SESSION_ID:
        chat = db.query(Chat).filter(Chat.id == session_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail=f"Чат с ID {session_id} не найден для очистки истории")

    try:
        # Удаляем сообщения из БД
        deleted_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(synchronize_session=False) # synchronize_session=False рекомендуется для bulk delete
        db.commit()

        # Очищаем память для этого чата
        if session_id in ai_memories:
            ai_memories[session_id].clear() # Очищаем содержимое памяти
            # Или удаляем ключ полностью, чтобы он пересоздался при след. сообщении
            # ai_memories.pop(session_id, None)
            logger.info(f"Память для чата {session_id} очищена.")

        logger.info(f"История чата {session_id} очищена. Удалено сообщений: {deleted_count}")
        return JSONResponse(content={"message": f"История чата очищена. Удалено сообщений: {deleted_count}"})

    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка очистки истории чата {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось очистить историю чата.")


# --- НОВАЯ Функция анализа изображения ---
async def analyze_image_with_ollama(
    image_bytes: bytes,
    prompt: str = IMAGE_ANALYSIS_PROMPT,
    model_name: str = VISION_MODEL_NAME
) -> Optional[str]:
    """
    Отправляет изображение и промпт в Ollama для анализа моделью Vision.
    """
    if not image_bytes:
        logger.error("analyze_image_with_ollama: Получены пустые байты изображения.")
        return None

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    ollama_api_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }
    logger.info(f"Отправка запроса на анализ изображения в Ollama моделью '{model_name}'...")

    try:
        # Используем httpx для простоты
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(ollama_api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if "response" in result:
                description = result["response"].strip()
                logger.info(f"Ollama Vision ({model_name}) успешно вернула описание.")
                logger.debug(f"Начало описания: {description[:200]}...")
                return description
            elif "error" in result:
                logger.error(f"Ollama вернула ошибку при анализе изображения: {result['error']}")
                return None
            else:
                logger.error(f"Неожиданный формат ответа от Ollama: {result}")
                return None
    except httpx.TimeoutException:
        logger.error(f"Таймаут при запросе к Ollama API ({ollama_api_url}).")
        return None
    except httpx.RequestError as e:
        logger.error(f"Ошибка сети при обращении к Ollama ({ollama_api_url}): {e}")
        return None
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при анализе изображения через Ollama: {e}", exc_info=True)
        return None