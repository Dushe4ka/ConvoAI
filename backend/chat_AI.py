import markdown
from fastapi import HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pydantic import BaseModel
import logging
import re

from database import get_db, ChatMessage
from qdrant_db import qdrant_client, collection_name
import uuid

# --- Настройки ---
logger = logging.getLogger(__name__)
llm = OllamaLLM(model="llama3.2")
vision_model = OllamaLLM(model="llama3.2-vision")
embed_model = OllamaEmbeddings(model="llama3.2")
ai_memories = {}

# --- Модель запроса ---
class ChatRequest(BaseModel):
    session_id: str
    message: str


def condense_question(user_message: str, memory: ConversationBufferMemory) -> str:
    """
    Конкретизация запроса: ИИ анализирует вопрос и формулирует его более чётко.
    """
    if len(user_message.split()) < 5:
        history = memory.load_memory_variables({}).get("history", "")
        prompt = (f"На основе следующего контекста:\n{history}\n"
                  f"Сформулируй уточняющий и конкретный запрос для поиска дополнительных данных по теме. "
                  f"Исходный вопрос: '{user_message}'")
        refined_question = llm.invoke(prompt)
        return refined_question.strip()
    return user_message


def generate_keywords(user_message: str, memory: ConversationBufferMemory) -> str:
    """
    Генерация ключевых слов на основе уточнённого запроса для поиска в базе данных Qdrant.
    """
    history = memory.load_memory_variables({}).get("history", "")
    prompt = (f"На основе следующего контекста:\n{history}\n"
              f"Исходный вопрос: '{user_message}'\n"
              f"Сформулируй ключевые слова для поиска в базе данных Qdrant и напиши их через запятую без лишнего текста")
    keywords = llm.invoke(prompt)
    return keywords.strip()


def format_message(message: str) -> str:
    """
    Преобразует текст, содержащий Markdown-разметку, в HTML.
    """
    return markdown.markdown(message)


def is_meaningless(text: str) -> bool:
    """Проверяет, является ли текст бессмысленным"""
    if re.fullmatch(r"[\d\s]+", text) or len(set(text)) == 1 or len(text) < 3:
        return True
    return False


def needs_more_context(text: str) -> bool:
    """Проверяет, слишком ли короткий вопрос"""
    return len(text.split()) < 3


# --- Генерация ответа ---
async def generate_rag_response(user_message: str, search_results: list, memory: ConversationBufferMemory):
    """
    Генерация финального ответа на основе пользовательского запроса и результатов поиска в Qdrant.
    """
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
        ai_response += "\n\n(Ответ сформирован на основе данных из хранилища)"
    else:
        ai_response += "\n\n(Ответ сгенерирован ИИ основываясь на его знаниях)"

    # Преобразуем итоговый ответ с Markdown-разметкой в HTML
    formatted_response = format_message(ai_response)
    return formatted_response


# --- Эндпоинт чата ---
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Пустой запрос")

    if is_meaningless(user_message):
        return JSONResponse(content={"response": "Попробуй сформулировать вопрос иначе."})

    if needs_more_context(user_message):
        return JSONResponse(content={"response": "Можешь уточнить свой вопрос?"})

    db.add(ChatMessage(session_id=session_id, role="user", message=user_message))
    db.commit()

    if session_id not in ai_memories:
        ai_memories[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    memory = ai_memories[session_id]

    # 1️⃣ Конкретизация запроса
    refined_query = condense_question(user_message, memory)
    if refined_query != user_message:
        logger.info(f"Вопрос преобразован из '{user_message}' в '{refined_query}' для поиска.")
    else:
        refined_query = user_message

    # 2️⃣ Генерация ключевых слов
    keywords = generate_keywords(refined_query, memory)
    logger.info(f"Ключевые слова для поиска: {keywords}")

    try:
        # 3️⃣ Поиск в Qdrant
        query_embedding = embed_model.embed_query(keywords)
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=10,
            score_threshold=0.7
        )
        if not search_results:
            logger.warning("В Qdrant не найдено релевантных данных.")

        # 4️⃣ Генерация финального ответа
        ai_response = await generate_rag_response(user_message, search_results, memory)

        # logger.info(f"Ключевые слова: {keywords}")
        logger.info(f"Вектор запроса: {query_embedding}")
        logger.info(f"Результаты поиска: {search_results}")

        db.add(ChatMessage(session_id=session_id, role="ai", message=ai_response))
        db.commit()

        memory.save_context({"input": user_message}, {"output": ai_response})

        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        logger.error(f"Ошибка в обработке запроса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сервера")


# --- Получение истории чатов ---
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    return [{"role": msg.role, "message": msg.message} for msg in history]


# --- Очистка истории чата ---
async def clear_chat_history(session_id: str, db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.commit()
    ai_memories.pop(session_id, None)
    return JSONResponse(content={"message": "История чата очищена"})


# --- Создание нового чата ---
async def create_new_chat():
    new_session_id = str(uuid.uuid4())
    return JSONResponse(content={"session_id": new_session_id})
