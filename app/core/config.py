from pydantic_settings import BaseSettings, SettingsConfigDict
from databases import Database
from sqlalchemy import create_engine, MetaData


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    SECRET_KEY: str
    SESSION_EXPIRE_HOURS: int = 24
    APP_NAME: str = "LLM Eval Platform"
    PWC_GENAI_API_KEY: str = ""
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
