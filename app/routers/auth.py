import uuid
from fastapi import APIRouter, HTTPException, Response, Depends
from pydantic import BaseModel, EmailStr
from app.core.config import database
from app.core.auth import hash_password, verify_password, create_session, delete_session, get_session_user

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    company: str | None = None
    role: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
async def register(body: RegisterRequest, response: Response):
    existing = await database.fetch_one("SELECT id FROM users WHERE email = :email", {"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    await database.execute(
        """
        INSERT INTO users (id, full_name, email, password_hash, company, role)
        VALUES (:id, :full_name, :email, :password_hash, :company, :role)
        """,
        {
            "id": user_id,
            "full_name": body.full_name,
            "email": body.email,
            "password_hash": hash_password(body.password),
            "company": body.company,
            "role": body.role,
        },
    )
    token = await create_session(user_id)
    response.set_cookie(
        key="session_token", value=token, httponly=True, samesite="lax", max_age=86400
    )
    return {"message": "Registered successfully", "user_id": user_id}


@router.post("/login")
async def login(body: LoginRequest, response: Response):
    user = await database.fetch_one(
        "SELECT id, password_hash, full_name, email, company, role FROM users WHERE email = :email",
        {"email": body.email},
    )
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = await create_session(str(user["id"]))
    response.set_cookie(
        key="session_token", value=token, httponly=True, samesite="lax", max_age=86400
    )
    return {
        "message": "Login successful",
        "user": {
            "id": str(user["id"]),
            "full_name": user["full_name"],
            "email": user["email"],
            "company": user["company"],
            "role": user["role"],
        },
    }


@router.post("/logout")
async def logout(response: Response, current_user=Depends(get_session_user)):
    await database.execute(
        "DELETE FROM sessions WHERE user_id = :uid", {"uid": current_user["user_id"]}
    )
    response.delete_cookie("session_token")
    return {"message": "Logged out"}


@router.get("/me")
async def me(current_user=Depends(get_session_user)):
    return current_user
