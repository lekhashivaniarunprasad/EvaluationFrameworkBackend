import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from databases import Database
from sqlalchemy import create_engine, MetaData

# load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    DB_NAME: str = "appdb"
    DB_USER: str = "admin"
    DB_PASSWORD: str = "admin"
    DB_HOST: str = "postgres-0.postgres-hl.default.svc.cluster.local"
    DB_PORT: int = 5432
    SECRET_KEY: str = "c06c0f853fe20c9c9082ba5f61e0e97bbecf444f7bb5372c8d8bc5dbdc8e3d52"
    SESSION_EXPIRE_HOURS: int = 24
    APP_NAME: str = "LLM Eval Platform"
    PWC_GENAI_API_KEY: str = "sk-SxXiWpNEB1MCA_yxD3eHiQ"
    PWC_GENAI_BASE_URL: str = "https://genai-sharedservice-americas.pwc.com"
    PWC_GENAI_MODEL: str = "vertex_ai.gemini-2.0-flash"
    PWC_GENAI_TIMEOUT_SECONDS: int = 300
    OPIK_PROJECT_NAME: str = "Default Project"
    # Arize Phoenix
    # PHOENIX_API_KEY:  str = "ak-c8469f94-e65b-4d10-b737-4edfcad54200-HjWsz1Alwe5GHnurOWIn5kgaZRW5bl1b"
    # PHOENIX_SPACE_ID: str = "U3BhY2U6NDA2MzY6TGV2Vg=="
    # # Arize AX
    ARIZE_SPACE_ID: str = "U3BhY2U6NDE5OTY6WGNIdg=="
    ARIZE_API_KEY:  str = "ak-c0c0f8d8-eac7-4523-b202-4f06dcf54bb5-T0gO3oizdlta382oU8Es06L7pEC5jRT2"
    # Opik
    OPIK_API_KEY: str = ""
    OPIK_WORKSPACE: str = ""

    def __init__(self, **values):
        env_values = {
            "DB_NAME": os.getenv("DB_NAME"),
            "DB_USER": os.getenv("DB_USER"),
            "DB_PASSWORD": os.getenv("DB_PASSWORD"),
            "DB_HOST": os.getenv("DB_HOST"),
            "DB_PORT": os.getenv("DB_PORT"),
            "SECRET_KEY": os.getenv("SECRET_KEY"),
            "SESSION_EXPIRE_HOURS": os.getenv("SESSION_EXPIRE_HOURS"),
            "APP_NAME": os.getenv("APP_NAME"),
            "PWC_GENAI_API_KEY": os.getenv("PWC_GENAI_API_KEY"),
            "PWC_GENAI_BASE_URL": os.getenv("PWC_GENAI_BASE_URL"),
            "PWC_GENAI_MODEL": os.getenv("PWC_GENAI_MODEL"),
            "PWC_GENAI_TIMEOUT_SECONDS": os.getenv("PWC_GENAI_TIMEOUT_SECONDS"),
            "OPIK_PROJECT_NAME": os.getenv("OPIK_PROJECT_NAME"),
            "ARIZE_SPACE_ID": os.getenv("ARIZE_SPACE_ID"),
            "ARIZE_API_KEY": os.getenv("ARIZE_API_KEY"),
            "OPIK_API_KEY": os.getenv("OPIK_API_KEY"),
            "OPIK_WORKSPACE": os.getenv("OPIK_WORKSPACE"),
        }
        env_values = {key: value for key, value in env_values.items() if value is not None}
        env_values.update(values)
        super().__init__(**env_values)

    @property
    def DATABASE_URL(self):
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def SYNC_DATABASE_URL(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


settings = Settings()

database = Database(settings.DATABASE_URL)
metadata = MetaData()
engine = create_engine(settings.SYNC_DATABASE_URL)
