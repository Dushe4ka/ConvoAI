from typing import Optional
from pydantic import BaseModel
from datetime import date


class TokenData(BaseModel):
    username: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str
    birth_date: date
    is_admin: bool = False


class UserProfileUpdate(BaseModel):
    password: Optional[str] = None
    full_name: Optional[str] = None
    birth_date: Optional[date] = None

class Token(BaseModel):
    access_token: str
    token_type: str


class UserBase(BaseModel):
    username: str
    full_name: str
    birth_date: date


class UserUpdate(BaseModel):
    password: Optional[str] = None
    full_name: Optional[str] = None
    birth_date: Optional[date] = None


class UserInDB(UserBase):
    id: int
    is_admin: bool

    class Config:
        orm_mode = True