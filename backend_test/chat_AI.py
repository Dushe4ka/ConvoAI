from fastapi import HTTPException, Depends, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pydantic import BaseModel
import logging
import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.parser import parse
import calendar

from backend_test.auth_utils import get_current_user
from database import get_db, ChatMessage, User
from qdrant_db import qdrant_client, collection_name

logger = logging.getLogger(__name__)
llm = OllamaLLM(model="llama3.2")
embed_model = OllamaEmbeddings(model="llama3.2")
ai_memories = {}

# Полная иерархия категорий
DEFAULT_CATEGORIES = {
    "Наука и технологии": {
        "Искусственный интеллект": ["Машинное обучение", "Нейросети", "Компьютерное зрение"],
        "Программирование": ["Веб-разработка", "Мобильные приложения", "Базы данных"],
        "Инженерия": ["Электроника", "Робототехника", "Строительство"]
    },
    "Бизнес и финансы": {
        "Финансы": ["Банки", "Инвестиции", "Криптовалюты"],
        "Маркетинг": ["SEO", "Контент-маркетинг", "Соцсети"],
        "Управление": ["Стартапы", "HR", "Проекты"]
    },
    "Образование": {
        "Школа": ["Учебники", "Методички", "ЕГЭ"],
        "ВУЗ": ["Лекции", "Курсовые", "Наука"],
        "Самообразование": ["Книги", "Онлайн-курсы", "Статьи"]
    },
    "Здоровье": {
        "Медицина": ["Болезни", "Лекарства", "Анализы"],
        "Спорт": ["Тренировки", "Питание", "Биомеханика"],
        "Психология": ["Отношения", "Стресс", "Мотивация"]
    },
    "Государство и право": {
        "Законы": ["Нормативные акты", "Кодексы", "Реформы"],
        "Недвижимость": ["Покупка", "Аренда", "Кадастр"],
        "Налоги": ["Отчетность", "Вычеты", "Проверки"]
    },
    "Творчество": {
        "Искусство": ["Живопись", "Музыка", "Фотография"],
        "Литература": ["Книги", "Поэзия", "Сценарии"],
        "Дизайн": ["Графика", "Интерьеры", "UI/UX"]
    },
    "Разное": {
        "История": ["События", "Личности", "Артефакты"],
        "Философия": ["Теории", "Этика", "Логика"],
        "Хобби": ["Путешествия", "Кулинария", "Гейминг"]
    }
}


class ChatRequest(BaseModel):
    session_id: str
    message: str


def extract_dates_and_category(text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Извлекает даты и категорию из текста запроса.
    Возвращает кортеж (даты, категория).
    """
    prompt = f"""
    Проанализируй текст и определи:
    1. Временной период (если указан)
    2. Основную категорию из списка: {list(DEFAULT_CATEGORIES.keys())}

    Формат ответа в JSON:
    {{
        "dates": {{
            "start": "YYYY-MM-DD" или null,
            "end": "YYYY-MM-DD" или null
        }},
        "category": "название категории" или null
    }}

    Примеры:
    1. "Новости о технологиях за 2023 год" → {{"dates": {{"start": "2023-01-01", "end": "2023-12-31"}}, "category": "Наука и технологии"}}
    2. "Последние исследования по медицине" → {{"dates": {{"start": "{datetime.now().replace(day=1).strftime('%Y-%m-%d')}", "end": null}}, "category": "Здоровье"}}
    3. "Любые документы о финансах" → {{"dates": null, "category": "Бизнес и финансы"}}

    Текст: "{text}"
    """

    try:
        response = llm.invoke(prompt)
        result = eval(response)  # Безопасный eval для простого JSON

        dates = None
        if result.get("dates"):
            dates = {
                "start": parse(result["dates"]["start"]).strftime('%Y-%m-%d') if result["dates"].get("start") else None,
                "end": parse(result["dates"]["end"]).strftime('%Y-%m-%d') if result["dates"].get("end") else None
            }

        category = result.get("category") if result.get("category") in DEFAULT_CATEGORIES else None

        return dates, category
    except Exception as e:
        logger.error(f"Ошибка анализа запроса: {e}")
        return None, None


def detect_subcategory(text: str, main_category: str) -> Optional[str]:
    """Определяет подкатегорию текста"""
    if main_category not in DEFAULT_CATEGORIES:
        return None

    prompt = f"""
    Определи подкатегорию текста из списка: {list(DEFAULT_CATEGORIES[main_category].keys())}
    Верни только название подкатегории. Текст: "{text[:300]}"
    """
    subcategory = llm.invoke(prompt).strip()
    return subcategory if subcategory in DEFAULT_CATEGORIES[main_category] else None


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


async def search_in_qdrant(
        query: str,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
        limit: int = 15
) -> List[Dict]:
    """Поиск в Qdrant с учетом всех параметров"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    must_conditions = []

    # Фильтр по категории
    if category:
        must_conditions.append(FieldCondition(
            key="category",
            match=MatchValue(value=category)
        ))

        # Фильтр по подкатегории
        if subcategory:
            must_conditions.append(FieldCondition(
                key="subcategory",
                match=MatchValue(value=subcategory)
            ))

    # Фильтр по датам
    if date_range:
        date_conditions = []
        if date_range.get("start"):
            date_conditions.append(Range(gte=date_range["start"]))
        if date_range.get("end"):
            date_conditions.append(Range(lte=date_range["end"]))

        if date_conditions:
            must_conditions.append(FieldCondition(
                key="date",
                range=Range(**{
                    k: v for cond in date_conditions
                    for k, v in cond.dict().items()
                })
            ))

    # Векторизация запроса
    query_embedding = embed_model.embed_query(query)

    # Выполнение поиска
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
        "date": hit.payload.get("date"),
        "category": hit.payload.get("category"),
        "subcategory": hit.payload.get("subcategory"),
        "score": hit.score
    } for hit in search_results]


async def generate_rag_response(
        query: str,
        context: List[Dict],
        dates: Optional[Dict[str, str]] = None,
        category: Optional[str] = None
) -> str:
    """Генерация ответа с учетом контекста"""
    context_str = "\n\n".join([
        f"🔹 Источник: {item['source']}\n"
        f"Дата: {item.get('date', 'неизвестно')}\n"
        f"Категория: {item.get('category', 'неизвестно')}"
        f"{' > ' + item['subcategory'] if item.get('subcategory') else ''}\n"
        f"Контент: {item['text'][:300]}..."
        for item in context
    ])

    date_info = ""
    if dates:
        start = dates.get('start', 'любое время')
        end = dates.get('end', 'любое время')
        date_info = f"\n\n📅 Поиск выполнен за период: {start} — {end}"

    category_info = f" в категории '{category}'" if category else ""

    prompt = f"""
    Ты — профессиональный аналитик. Ответь на вопрос, используя ТОЛЬКО предоставленные данные.
    {date_info}
    {f"Категория запроса: {category_info}" if category_info else ""}

    ===== ДАННЫЕ =====
    {context_str if context else "Нет релевантных данных"}

    ===== ВОПРОС =====
    {query}

    ===== ТРЕБОВАНИЯ К ОТВЕТУ =====
    1. Начни с краткого вывода (1-2 предложения)
    2. Приведи 3-5 ключевых фактов/цифр
    3. Укажи источники в формате "[1] filename.pdf"
    4. Если данных недостаточно — честно скажи об этом
    5. Сохраняй профессиональный тон

    Ответ:
    """
    return llm.invoke(prompt)


async def chat_endpoint(
        request: ChatRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    session_id = request.session_id
    user_message = request.message.strip()

    # Валидация запроса
    if not user_message or len(user_message) < 3:
        raise HTTPException(status_code=400, detail="Слишком короткий запрос")

    # Сохранение сообщения пользователя с привязкой к пользователю
    db.add(ChatMessage(
        session_id=session_id,
        role="user",
        message=user_message,
        user_id=current_user.id
    ))
    db.commit()

    # Инициализация памяти диалога
    if session_id not in ai_memories:
        ai_memories[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    memory = ai_memories[session_id]

    try:
        # 1. Анализ запроса
        refined_query = condense_question(user_message, memory)
        dates, category = extract_dates_and_category(refined_query)
        subcategory = detect_subcategory(refined_query, category) if category else None
        keywords = generate_search_keywords(refined_query, category)

        logger.info(
            f"Поиск: '{refined_query}' | "
            f"Категория: {category}{' > ' + subcategory if subcategory else ''} | "
            f"Даты: {dates} | "
            f"Ключевые слова: {keywords}"
        )

        # 2. Поиск в хранилище
        search_results = await search_in_qdrant(
            query=keywords,
            category=category,
            subcategory=subcategory,
            date_range=dates,
            limit=15
        )

        # 3. Генерация ответа
        ai_response = await generate_rag_response(
            query=user_message,
            context=search_results,
            dates=dates,
            category=category
        )

        # 4. Проверка и сохранение
        if len(ai_response) > 10000:  # Защита от слишком длинных ответов
            ai_response = ai_response[:10000] + "\n\n[Ответ сокращен]"

        db.add(ChatMessage(
            session_id=session_id,
            role="ai",
            message=ai_response,
            user_id=current_user.id
        ))
        db.commit()
        memory.save_context({"input": user_message}, {"output": ai_response})

        return JSONResponse(content={
            "response": ai_response,
            "metadata": {
                "category": category,
                "subcategory": subcategory,
                "dates": dates,
                "sources": list(set(res['source'] for res in search_results)),
                "user_id": current_user.id
            }
        })

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


async def get_chat_history(
        session_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # Получаем историю только для текущего пользователя
    history = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id,
        ChatMessage.user_id == current_user.id
    ).order_by(ChatMessage.timestamp.asc()).all()

    return [{
        "role": msg.role,
        "message": msg.message,
        "timestamp": msg.timestamp.isoformat()
    } for msg in history]


async def clear_chat_history(
        session_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # Удаляем историю только для текущего пользователя
    deleted_count = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id,
        ChatMessage.user_id == current_user.id
    ).delete()
    db.commit()

    if deleted_count > 0:
        ai_memories.pop(session_id, None)

    return JSONResponse(content={
        "message": "История чата очищена",
        "deleted_messages": deleted_count,
        "session_id": session_id
    })


async def create_new_chat(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_session_id = str(uuid.uuid4())

    # Создаем первую системную запись в чате
    db.add(ChatMessage(
        session_id=new_session_id,
        role="system",
        message="Новый чат создан",
        user_id=current_user.id
    ))
    db.commit()

    return JSONResponse(content={
        "session_id": new_session_id,
        "user_id": current_user.id,
        "created_at": datetime.utcnow().isoformat()
    })


async def get_categories():
    """Возвращает иерархию категорий"""
    return JSONResponse(content=DEFAULT_CATEGORIES)