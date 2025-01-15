# app/logging_config.py
# This file sets up logging to track system operations and errors
import logging
import sys
from pathlib import Path

def setup_logging():
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# app/health.py
# This file adds system health monitoring
from typing import Dict
import psutil
import os
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

class HealthCheck:
    def __init__(self, db_url: str, qdrant_host: str):
        self.db_url = db_url
        self.qdrant_host = qdrant_host
        self.engine = create_engine(db_url)
        self.qdrant = QdrantClient(host=qdrant_host)

    async def check_postgres(self) -> Dict:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_qdrant(self) -> Dict:
        try:
            self.qdrant.get_collections()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_system_resources(self) -> Dict:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }

# app/main.py
# Updated main application with health checks and logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .database import Database
from .embeddings import Embeddings
from .models import Query, Response
from .health import HealthCheck
from .logging_config import setup_logging
import os
import json
from typing import Dict

# Initialize logging
logger = setup_logging()

app = FastAPI(title="Enhanced RAG API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = Database()
embeddings = Embeddings(settings.model_name)
health_checker = HealthCheck(
    db_url=f"postgresql://{settings.postgres_user}:{os.getenv('POSTGRES_PASSWORD')}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
    qdrant_host=os.getenv("QDRANT_HOST", "localhost")
)

@app.get("/health")
async def health_check() -> Dict:
    """
    Comprehensive system health check endpoint
    """
    return {
        "postgres": await health_checker.check_postgres(),
        "qdrant": await health_checker.check_qdrant(),
        "system": await health_checker.check_system_resources()
    }

@app.post("/documents/upload")
async def upload_documents(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Enhanced document upload endpoint with background processing
    """
    try:
        logger.info(f"Starting upload for file: {file.filename}")
        
        # Save file temporarily
        content = await file.read()
        
        # Store in PostgreSQL
        with db.SessionLocal() as session:
            session.execute(
                text("INSERT INTO documents (content, metadata) VALUES (:content, :metadata)"),
                {
                    "content": content.decode(),
                    "metadata": json.dumps({
                        "filename": file.filename,
                        "size": len(content),
                        "type": file.content_type
                    })
                }
            )
            session.commit()
            logger.info("Document metadata stored in PostgreSQL")
        
        # Process for vector search in background
        def process_vectors():
            try:
                chunks = chunk_text(content.decode(), 
                                  settings.chunk_size, 
                                  settings.chunk_overlap)
                vectors = embeddings.encode(chunks)
                db.qdrant.upsert(
                    collection_name=settings.collection_name,
                    points=[
                        {"id": i, "vector": vec.tolist(), "payload": {"text": chunk}}
                        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
                    ]
                )
                logger.info(f"Vector processing completed for {file.filename}")
            except Exception as e:
                logger.error(f"Vector processing failed: {str(e)}")
        
        if background_tasks:
            background_tasks.add_task(process_vectors)
        else:
            process_vectors()
        
        return {
            "message": "Document uploaded successfully",
            "status": "processing" if background_tasks else "completed"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Response)
async def query(query: Query):
    """
    Enhanced query endpoint with better error handling
    """
    try:
        logger.info(f"Processing query: {query.text[:50]}...")
        
        # Create query embedding
        query_vector = embeddings.encode_query(query.text)
        
        # Search for similar documents
        results = db.qdrant.search(
            collection_name=settings.collection_name,
            query_vector=query_vector.tolist(),
            limit=query.top_k
        )
        
        if not results:
            logger.warning("No results found for query")
            return Response(
                answer="No relevant documents found",
                sources=[]
            )
        
        # Extract text from results
        sources = [result.payload["text"] for result in results]
        logger.info(f"Found {len(sources)} relevant documents")
        
        return Response(
            answer=f"Found {len(sources)} relevant documents",
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008, reload=True)