from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import database
from app.core.db_init import create_tables
from app.routers import auth, projects, evaluate

app = FastAPI(title="LLM Evaluation Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    create_tables()       # auto-create tables if they don't exist
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(evaluate.router)


@app.get("/")
async def root():
    return {"message": "Backend is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
