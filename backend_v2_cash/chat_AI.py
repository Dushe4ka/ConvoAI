from fastapi import HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pydantic import BaseModel
import logging
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dateutil.parser import parse

from database import get_db, ChatMessage
from qdrant_db import qdrant_client, collection_name

logger = logging.getLogger(__name__)
llm = OllamaLLM(model="llama3.2")
embed_model = OllamaEmbeddings(model="llama3.2")
ai_memories = {}

CATEGORIES = [
    "Наука и технологии",
    "Бизнес и финансы",
    "Образование",
    "Здоровье",
    "Государство и право",
    "Творчество",
    "Разное"
]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    filters: Optional[dict] = None


class SearchFilters(BaseModel):
    category: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


def extract_search_params(text: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Извлекает категорию и даты из текста запроса"""
    prompt = f"""
    Определи категорию документа из списка: {CATEGORIES}
    Если в тексте указан период, определи даты в формате YYYY-MM-DD.
    Формат ответа JSON: {{"category": "...", "dates": {{"start": "...", "end": "..."}}}}
    Текст: "{text[:500]}"
    """
    try:
        response = llm.invoke(prompt)
        result = eval(response)

        category = result.get("category") if result.get("category") in CATEGORIES else None

        dates = None
        if result.get("dates"):
            dates = {
                "start": parse(result["dates"]["start"]).strftime('%Y-%m-%d') if result["dates"].get("start") else None,
                "end": parse(result["dates"]["end"]).strftime('%Y-%m-%d') if result["dates"].get("end") else None
            }

        return category, dates
    except Exception as e:
        logger.error(f"Ошибка анализа запроса: {e}")
        return None, None


async def search_in_qdrant(
        query: str,
        category: Optional[str] = None,
        upload_dates: Optional[Dict[str, str]] = None,
        limit: int = 15
) -> List[Dict]:
    """Поиск документов с фильтрацией"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    must_conditions = []

    if category:
        must_conditions.append(FieldCondition(
            key="category",
            match=MatchValue(value=category)
        ))

    if upload_dates:
        date_conditions = []
        if upload_dates.get("start"):
            date_conditions.append(Range(gte=upload_dates["start"]))
        if upload_dates.get("end"):
            date_conditions.append(Range(lte=upload_dates["end"]))

        if date_conditions:
            must_conditions.append(FieldCondition(
                key="upload_date",
                range=Range(**{
                    k: v for cond in date_conditions
                    for k, v in cond.dict().items()
                })
            ))

    query_embedding = embed_model.embed_query(query)

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=Filter(must=must_conditions) if must_conditions else None,
        limit=limit,
        with_payload=True
    )

    return [{
        "text": hit.payload["text"],
        "source": hit.payload.get("source", "unknown"),
        "upload_date": hit.payload.get("upload_date"),
        "category": hit.payload.get("category"),
        "score": hit.score
    } for hit in search_results]


async def generate_rag_response(
        query: str,
        context: List[Dict],
        category: Optional[str] = None,
        upload_dates: Optional[Dict[str, str]] = None
) -> str:
    """Генерация ответа с учетом контекста"""
    context_str = "\n\n".join([
        f"🔹 Источник: {item['source']}\n"
        f"Дата загрузки: {item.get('upload_date', 'неизвестно')}\n"
        f"Категория: {item.get('category', 'неизвестно')}\n"
        f"Контент: {item['text'][:300]}..."
        for item in context
    ])

    filters_info = []
    if category:
        filters_info.append(f"Категория: {category}")
    if upload_dates:
        start = upload_dates.get('start', 'любая')
        end = upload_dates.get('end', 'любая')
        filters_info.append(f"Период загрузки: {start} — {end}")

    prompt = f"""
    Ответь на вопрос, используя только предоставленные данные.
    {f"Фильтры поиска: {', '.join(filters_info)}" if filters_info else ""}

    Данные:
    {context_str if context else "Нет релевантных данных"}

    Вопрос: {query}

    Требования к ответу:
    1. Будь точным и лаконичным
    2. Укажи источники в формате [1] filename.pdf
    3. Если данных недостаточно — сообщи об этом
    """
    return llm.invoke(prompt).strip()


async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message or len(user_message) < 3:
        raise HTTPException(status_code=400, detail="Слишком короткий запрос")

    db.add(ChatMessage(session_id=session_id, role="user", message=user_message))
    db.commit()

    if session_id not in ai_memories:
        ai_memories[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    memory = ai_memories[session_id]

    try:
        # Обработка фильтров
        filters = request.filters or {}
        category = filters.get("category")
        upload_dates = None

        if filters.get("date_from") or filters.get("date_to"):
            upload_dates = {
                "start": filters.get("date_from"),
                "end": filters.get("date_to")
            }

        refined_query = condense_question(user_message, memory)

        # Если категория не указана в фильтрах, определяем автоматически
        if not category:
            category, _ = extract_search_params(refined_query)

        keywords = generate_search_keywords(refined_query, category)

        logger.info(f"Поиск: '{refined_query}' | Категория: {category} | Даты: {upload_dates}")

        search_results = await search_in_qdrant(
            query=keywords,
            category=category,
            upload_dates=upload_dates,
            limit=15
        )

        ai_response = await generate_rag_response(
            query=user_message,
            context=search_results,
            category=category,
            upload_dates=upload_dates
        )

        if len(ai_response) > 10000:
            ai_response = ai_response[:10000] + "\n\n[Ответ сокращен]"

        metadata = {
            "category": category,
            "upload_dates": upload_dates,
            "sources": list(set(res['source'] for res in search_results))
        }

        db.add(ChatMessage(session_id=session_id, role="ai", message=ai_response))
        db.commit()
        memory.save_context({"input": user_message}, {"output": ai_response})

        return JSONResponse(content={
            "response": ai_response,
            "metadata": metadata
        })

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    return [{"role": msg.role, "message": msg.message} for msg in history]


async def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.commit()
    ai_memories.pop(session_id, None)
    return JSONResponse(content={"message": "История чата очищена"})


async def create_new_chat():
    new_session_id = str(uuid.uuid4())
    return JSONResponse(content={"session_id": new_session_id})


def generate_search_keywords(question: str, category: Optional[str] = None) -> str:
    """Генерирует ключевые слова для поиска с учетом категории"""
    prompt = f"""
    Извлеки ключевые слова из вопроса для поиска в документах. 
    {"Учти, что документы относятся к категории: " + category + "." if category else ""}
    Формат: слова через запятую, на русском языке.

    Вопрос: "{question}"

    Ключевые слова:
    """
    return llm.invoke(prompt).strip()


def condense_question(user_message: str, memory: ConversationBufferMemory) -> str:
    """Уточняет и улучшает формулировку вопроса"""
    history = memory.load_memory_variables({}).get("history", "")
    prompt = f"""
    Улучши формулировку вопроса для более точного поиска. Сохрани исходный смысл.

    История диалога:
    {history}

    Исходный вопрос:
    "{user_message}"

    Улучшенный вопрос:
    """
    return llm.invoke(prompt).strip()