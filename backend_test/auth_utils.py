# auth_utils.py

from datetime import datetime, timedelta, timezone # Импортируем timezone
from typing import Optional # Импортируем Optional для type hinting

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# Подразумевается, что эти импорты работают и указывают на ваши модели/сессии
from database import get_db, User

# --- Конфигурация ---
# Лучше вынести в отдельный файл конфигурации или переменные окружения
# Убедитесь, что эти значения СООТВЕТСТВУЮТ тем, что могут быть в app.py
SECRET_KEY = "your-secret-key-here"  # !!! ЗАМЕНИТЕ НА ВАШ НАДЕЖНЫЙ КЛЮЧ !!!
ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES больше не используется напрямую в create_access_token,
# но может использоваться при вызове функции в app.py
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Определение схемы OAuth2 - может быть определено здесь или в app.py
# Убедитесь, что используется только одно определение
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Контекст для работы с паролями
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --- Функции для работы с паролями ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет, соответствует ли обычный пароль хешированному."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Генерирует хеш для пароля."""
    return pwd_context.hash(password)


# --- Функции для работы с JWT токенами ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Создает JWT токен доступа.

    Args:
        data: Словарь с данными для включения в токен (например, {'sub': username}).
        expires_delta: Время жизни токена (объект timedelta). Если None,
                       используется значение по умолчанию (например, 15 минут).

    Returns:
        Сгенерированный JWT токен в виде строки.
    """
    to_encode = data.copy()
    if expires_delta:
        # Используем переданное время жизни
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Используем время жизни по умолчанию, если не передано
        expire = datetime.now(timezone.utc) + timedelta(minutes=15) # Значение по умолчанию - 15 минут

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# --- Функции-зависимости для FastAPI ---

async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
) -> User:
    """
    Декодирует токен, проверяет его и возвращает пользователя из БД.
    Используется как зависимость FastAPI для защищенных эндпоинтов.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            print("Username not found in token payload") # Отладка
            raise credentials_exception
    except JWTError as e:
        print(f"JWT Decoding Error: {e}") # Отладка
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        print(f"User '{username}' not found in database") # Отладка
        raise credentials_exception
    return user


async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    """
    Проверяет, имеет ли текущий пользователь права администратора.
    Используется как зависимость FastAPI для админских эндпоинтов.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user
