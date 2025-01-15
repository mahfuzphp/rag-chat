
# File: app/config.py
# Purpose: Application configuration settings
from pydantic import BaseSettings

class Settings(BaseSettings):
    model_name: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int

    class Config:
        env_file = ".env"

settings = Settings()