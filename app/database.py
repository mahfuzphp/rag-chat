
# app/database.py
# Updated database handler with port configuration
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
import time

class Database:
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=6333
        )
        
        # PostgreSQL connection setup with retry logic
        self.init_postgres_connection()

    def init_postgres_connection(self, max_retries=5):
        """Initialize PostgreSQL connection with retry logic"""
        for attempt in range(max_retries):
            try:
                # Build connection URL using environment variables
                postgres_url = (
                    f"postgresql://{os.getenv('POSTGRES_USER')}:"
                    f"{os.getenv('POSTGRES_PASSWORD')}@"
                    f"{os.getenv('POSTGRES_HOST')}:"
                    f"{os.getenv('POSTGRES_PORT')}/"
                    f"{os.getenv('POSTGRES_DB')}"
                )
                
                # Create engine and test connection
                self.engine = create_engine(postgres_url)
                self.SessionLocal = sessionmaker(bind=self.engine)
                
                # Test the connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                print("Successfully connected to PostgreSQL")
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to PostgreSQL after {max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                time.sleep(5)

    def init_collections(self, collection_name: str, vector_size: int):
        """Initialize both Qdrant and PostgreSQL storage"""
        # Initialize Qdrant collection
        try:
            self.qdrant.get_collection(collection_name)
        except:
            self.qdrant.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
        # Initialize PostgreSQL tables with schema
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    chunk_index INTEGER,
                    vector_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))