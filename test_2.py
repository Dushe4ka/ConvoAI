from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Distance, PointStruct
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uuid
import logging
import os
import fitz
from docx import Document
import markdown

# Конфигурация
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
COLLECTION_NAME = "main_documents"
UPLOAD_DIR = "uploads"

# Инициализация
Base = declarative_base()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# База данных
DATABASE_URL = "sqlite:///./chat.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
embed_model = OllamaEmbeddings(model="llama3.2")
llm = OllamaLLM(model="llama3.2")


# Модели Pydantic
class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr | None = None


class Token(BaseModel):
    access_token: str
    token_type: str


class ChatMessageRequest(BaseModel):
    message: str


class ChatSessionResponse(BaseModel):
    id: int
    name: str
    session_id: str
    created_at: str


# Модели базы данных
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, nullable=True)
    hashed_password = Column(String)
    created_at = Column(String, default=datetime.utcnow().isoformat())
    chats = relationship("ChatSession", back_populates="owner")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(String, default=datetime.utcnow().isoformat())
    owner = relationship("User", back_populates="chats")
    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)
    message = Column(Text)
    timestamp = Column(String, default=datetime.utcnow().isoformat())
    session = relationship("ChatSession", back_populates="messages")


Base.metadata.create_all(bind=engine)

# Аутентификация
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


# Инициализация Qdrant
def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections()
        if not any(col.name == COLLECTION_NAME for col in collections.collections):
            test_vector = embed_model.embed_query("test")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(test_vector), distance=Distance.COSINE),
            )
            logger.info(f"Created collection {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Qdrant initialization error: {e}")
        raise


initialize_qdrant()


# Вспомогательные функции
def format_message(message: str) -> str:
    return markdown.markdown(message)


ai_memories = {}


# Эндпоинты
@app.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return {
        "access_token": create_access_token({"sub": user.username}),
        "token_type": "bearer"
    }


@app.post("/chats", response_model=ChatSessionResponse)
async def create_chat_session(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    session_id = str(uuid.uuid4())
    new_session = ChatSession(
        user_id=current_user.id,
        name=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        session_id=session_id
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@app.get("/chats", response_model=list[ChatSessionResponse])
async def get_chat_sessions(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    return db.query(ChatSession).filter(ChatSession.user_id == current_user.id).all()


@app.post("/chat/{session_id}")
async def chat(
        session_id: str,
        request: ChatMessageRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    # Проверка доступа к сессии
    chat_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    # Сохранение сообщения пользователя
    db_message = ChatMessage(
        session_id=chat_session.id,
        role="user",
        message=user_message
    )
    db.add(db_message)

    # Инициализация памяти
    if session_id not in ai_memories:
        ai_memories[session_id] = ConversationBufferMemory()
    memory = ai_memories[session_id]

    try:
        # Поиск в Qdrant с фильтром по пользователю
        query_embedding = embed_model.embed_query(user_message)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=current_user.id))]
            ),
            limit=3
        )

        # Формирование контекста
        context = "\n".join([hit.payload["text"] for hit in search_results if hit.score > 0.7])
        prompt = f"Context: {context}\n\nQuestion: {user_message}"

        # Генерация ответа
        ai_response = llm.invoke(prompt)
        formatted_response = format_message(ai_response)

        # Сохранение ответа ИИ
        db_ai_message = ChatMessage(
            session_id=chat_session.id,
            role="ai",
            message=formatted_response
        )
        db.add(db_ai_message)
        db.commit()

        # Обновление памяти
        memory.save_context({"input": user_message}, {"output": ai_response})

        return JSONResponse(content={"response": formatted_response})

    except Exception as e:
        db.rollback()
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")


@app.delete("/chat/{session_id}")
async def delete_chat(
        session_id: str,
        current_user: User = Depends(get_current_user)),


        db: Session = Depends(get_db)
        ):
        # Удаление сессии чата
        chat_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
        ).first()
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        try:
            # Удаление связанных данных
            db.query(ChatMessage).filter(ChatMessage.session_id == chat_session.id).delete()
            db.delete(chat_session)

            # Удаление векторов пользователя из Qdrant
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=current_user.id))]
                )
            )

            # Очистка памяти
            if session_id in ai_memories:
                del ai_memories[session_id]

            db.commit()
            return PlainTextResponse("Chat deleted successfully")

        except Exception as e:
            db.rollback()
            logger.error(f"Delete error: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete chat")


@app.post("/upload")
async def upload_file(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)),


db: Session = Depends(get_db)
):
try:
    # Сохранение файла
    file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Обработка файла
    text = ""
    if file.filename.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
    elif file.filename.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(file_path, "r") as f:
            text = f.read()

    # Разделение текста и сохранение в Qdrant
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    points = []
    for chunk in chunks:
        embedding = embed_model.embed_query(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "user_id": current_user.id,
                "source": file.filename
            }
        ))

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return JSONResponse(content={"message": "File processed successfully"})

except Exception as e:
    logger.error(f"Upload error: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)