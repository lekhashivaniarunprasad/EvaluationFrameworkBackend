import uuid
import secrets
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from fastapi import HTTPException, Cookie, Depends
from app.core.config import database, settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


async def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_EXPIRE_HOURS)
    await database.execute(
        """
        INSERT INTO sessions (id, user_id, session_token, expires_at)
        VALUES (:id, :user_id, :token, :expires_at)
        """,
        {"id": str(uuid.uuid4()), "user_id": user_id, "token": token, "expires_at": expires_at},
    )
    return token


async def get_session_user(session_token: str = Cookie(default=None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    row = await database.fetch_one(
        """
        SELECT s.user_id, s.expires_at, u.full_name, u.email, u.company, u.role
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.session_token = :token
        """,
        {"token": session_token},
    )
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    if row["expires_at"] < datetime.now(timezone.utc):
        await database.execute("DELETE FROM sessions WHERE session_token = :token", {"token": session_token})
        raise HTTPException(status_code=401, detail="Session expired")
    return dict(row)


async def delete_session(session_token: str):
    await database.execute("DELETE FROM sessions WHERE session_token = :token", {"token": session_token})
