# app/config.py
# Configuration settings with environment awareness
from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    model_name: str = "microsoft/phi-2"
    collection_name: str = "documents"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Database settings
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5434")  # Default to the new port
    postgres_db: str = os.getenv("POSTGRES_DB", "ragdb")
    postgres_user: str = os.getenv("POSTGRES_USER", "raguser")
    
    # Data directories
    data_dir: str = "/data"
    documents_dir: str = os.path.join(data_dir, "documents")

    class Config:
        env_file = ".env"

settings = Settings()