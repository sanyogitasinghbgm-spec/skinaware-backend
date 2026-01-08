from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
import os

pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"

from fastapi import HTTPException

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto"
)
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



def hash_password(password):
    return pwd_context.hash(password)

# def hash_password(password: str) -> str:
#     if len(password.encode("utf-8")) > 72:
#         raise ValueError("Password too long (max 72 characters)")
#     return pwd_context.hash(password)


def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(user_id):
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(hours=8)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
