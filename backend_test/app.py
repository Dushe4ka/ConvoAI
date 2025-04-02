from fastapi import FastAPI, Request, Depends, HTTPException, status, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Generator
import logging
import os
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.orm import Session  # Добавьте этот импорт

# Импорт модулей проекта
from file_handler import upload_file, get_categories
from chat_AI import chat_endpoint, get_chat_history, clear_chat_history, create_new_chat
from database import Base, engine, get_db, User, SessionLocal  # Добавьте SessionLocal
from qdrant_db import initialize_qdrant
from models import TokenData, UserCreate, UserProfileUpdate
from auth_utils import create_access_token, verify_password, get_password_hash


# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация приложения
app = FastAPI(
    title="AI Knowledge Base",
    description="Система анализа документов с RAG-поиском и аутентификацией",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация аутентификации
SECRET_KEY = "your-secret-key-here"  # В продакшене заменить на надежный ключ
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Подключение статических файлов
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / ".." / "frontend_test" / "static"
TEMPLATES_DIR = BASE_DIR / ".." / "frontend_test" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Инициализация БД
Base.metadata.create_all(bind=engine)
initialize_qdrant()


# Модели для аутентификации
class Token(BaseModel):
    access_token: str
    token_type: str


class UserLogin(BaseModel):
    username: str
    password: str


# Функции аутентификации
async def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_admin(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Эндпоинты аутентификации
@app.post("/api/auth/login", response_model=Token)
async def login_for_access_token(
        form_data: UserLogin,
        db: Session = Depends(get_db)
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/auth/register")
async def register_user(
        user_data: UserCreate,
        db: Session = Depends(get_db),
        admin: User = Depends(get_current_active_admin)
):
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        birth_date=user_data.birth_date,
        is_admin=user_data.is_admin
    )
    db.add(db_user)
    db.commit()
    return {"message": "User created successfully"}


# Эндпоинты профиля
@app.get("/api/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "full_name": current_user.full_name,
        "birth_date": current_user.birth_date,
        "is_admin": current_user.is_admin
    }


@app.put("/api/profile")
async def update_user_profile(
        update_data: UserProfileUpdate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if update_data.password:
        current_user.hashed_password = get_password_hash(update_data.password)
    if update_data.full_name:
        current_user.full_name = update_data.full_name
    if update_data.birth_date:
        current_user.birth_date = update_data.birth_date

    db.commit()
    return {"message": "Profile updated successfully"}


# Защищенные эндпоинты API
@app.post("/api/chat")
async def protected_chat_endpoint(
        request: Request,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    return await chat_endpoint(request, db, current_user)


@app.get("/api/history/{session_id}")
async def protected_get_chat_history(
        session_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    return await get_chat_history(session_id, db, current_user)


@app.delete("/api/chat/{session_id}")
async def protected_clear_chat_history(
        session_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    return await clear_chat_history(session_id, db, current_user)


@app.post("/api/upload")
async def protected_upload_file(
        file: UploadFile,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    return await upload_file(file, db, current_user)


# Публичные эндпоинты
@app.get("/api/categories")
async def public_get_categories():
    return await get_categories()


@app.get("/api/new_chat")
async def public_create_new_chat():
    return await create_new_chat()


# Админские эндпоинты
@app.get("/api/admin/users")
async def admin_list_users(
        db: Session = Depends(get_db),
        admin: User = Depends(get_current_active_admin)
):
    return db.query(User).all()


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
        user_id: int,
        db: Session = Depends(get_db),
        admin: User = Depends(get_current_active_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}


# Основной маршрут
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Обработчики событий
@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    db = SessionLocal()
    try:
        # Создаем администратора по умолчанию
        if not db.query(User).filter(User.username == "admin").first():
            admin = User(
                username="admin",
                hashed_password=get_password_hash("admin"),
                full_name="Admin",
                birth_date=datetime.now().date(),
                is_admin=True
            )
            db.add(admin)
            db.commit()
            logger.info("Default admin user created")

        # Проверяем соединение с Qdrant
        initialize_qdrant()
        logger.info("Qdrant connection established")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        db.close()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")


# Запуск приложения
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        workers=4
    )