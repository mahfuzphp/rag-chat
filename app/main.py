# Main FastAPI application file implementing a storage-optimized RAG system
# This file handles document processing with compression and efficient resource usage

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import json
import os
import zlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Import our application components
from config import settings
from database import Database
from models import Query, Response, Document, SearchResult
from embeddings import Embeddings
from text_processor import TextProcessor
from utils import setup_logging

# Initialize logging with rotation to manage file sizes
logger = setup_logging(
    max_bytes=10485760,  # 10MB
    backup_count=5
)

# Create FastAPI application with minimal startup load
app = FastAPI(
    title="Storage-Optimized RAG System",
    description="A space-efficient RAG system for document processing and querying",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with storage optimization
db = Database()
embeddings = Embeddings(
    model_name=settings.model_name,
    batch_size=32  # Process embeddings in smaller batches
)
text_processor = TextProcessor(
    chunk_size=256,  # Smaller chunks for storage efficiency
    chunk_overlap=20  # Reduced overlap
)

# Cleanup configuration
RETENTION_DAYS = 30
CLEANUP_BATCH_SIZE = 100

@app.on_event("startup")
async def startup_event():
    """Initialize system components with storage optimization"""
    logger.info("Starting storage-optimized RAG system...")
    try:
        await db.initialize()
        logger.info("Database connections established")
        
        # Create minimal required directories
        if not os.path.exists(settings.documents_dir):
            os.makedirs(settings.documents_dir)
            logger.info("Document directory initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources and compress logs on shutdown"""
    logger.info("Performing cleanup before shutdown...")
    try:
        await db.close_connections()
        await compress_old_logs()
    except Exception as e:
        logger.error(f"Shutdown cleanup failed: {str(e)}")

async def compress_old_logs():
    """Compress log files older than 1 day"""
    log_dir = "logs"
    current_time = datetime.now()
    
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            file_path = os.path.join(log_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if (current_time - file_time) > timedelta(days=1):
                with open(file_path, 'rb') as f_in:
                    with open(f"{file_path}.gz", 'wb') as f_out:
                        f_out.write(zlib.compress(f_in.read()))
                os.remove(file_path)

async def cleanup_old_documents():
    """Remove documents older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
    
    while True:
        deleted_count = await db.delete_old_documents(
            cutoff_date,
            batch_size=CLEANUP_BATCH_SIZE
        )
        if deleted_count < CLEANUP_BATCH_SIZE:
            break

@app.get("/health")
async def health_check() -> Dict:
    """Check system health and storage usage"""
    try:
        disk_usage = os.statvfs(settings.documents_dir)
        free_space = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024 * 1024 * 1024)  # GB
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "storage": {
                "free_space_gb": round(free_space, 2),
                "storage_status": "ok" if free_space > 1 else "low"
            },
            "components": {
                "database": await db.check_health(),
                "vector_store": await db.check_vector_store(),
                "embeddings": embeddings.check_health()
            }
        }
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/documents/upload")
async def upload_documents(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict:
    """Upload and process documents with storage optimization"""
    try:
        logger.info(f"Processing upload: {file.filename}")
        
        # Read and compress content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        compressed_content = zlib.compress(content)
        
        # Store compressed document metadata
        doc_id = await db.store_document_metadata({
            "filename": file.filename,
            "original_size": len(content),
            "compressed_size": len(compressed_content),
            "upload_time": datetime.now().isoformat(),
            "content_type": file.content_type,
            "compressed_content": compressed_content
        })
        
        # Process document efficiently
        async def process_document():
            try:
                text_content = zlib.decompress(compressed_content).decode('utf-8')
                chunks = text_processor.chunk_text(text_content)
                
                # Process embeddings in batches
                for i in range(0, len(chunks), embeddings.batch_size):
                    batch = chunks[i:i + embeddings.batch_size]
                    batch_embeddings = embeddings.encode_batch(batch)
                    await db.store_chunks(doc_id, batch, batch_embeddings)
                
                logger.info(f"Successfully processed document {file.filename}")
                
                # Cleanup old documents if needed
                background_tasks.add_task(cleanup_old_documents)
                
            except Exception as e:
                logger.error(f"Background processing failed for {file.filename}: {str(e)}")
                await db.mark_document_failed(doc_id, str(e))
        
        if background_tasks:
            background_tasks.add_task(process_document)
            status = "processing"
        else:
            await process_document()
            status = "completed"
            
        return {
            "message": f"Document upload {status}",
            "document_id": doc_id,
            "status": status,
            "storage_saved": round((len(content) - len(compressed_content)) / 1024, 2)  # KB saved
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query) -> Response:
    """Query the RAG system with optimized search"""
    try:
        logger.info(f"Processing query: {query.text[:100]}...")
        query_embedding = embeddings.encode_text(query.text)
        
        results = await db.semantic_search(
            query_embedding,
            limit=min(query.top_k or 5, 10),  # Limit max results for efficiency
            threshold=query.threshold or 0.7
        )
        
        if not results:
            logger.warning("No relevant results found")
            return Response(
                answer="No relevant documents found",
                sources=[],
                confidence=0.0
            )
        
        sources = [
            SearchResult(
                text=result.text,
                confidence=result.score,
                document_id=result.document_id,
                metadata=result.metadata
            )
            for result in results
        ]
        
        avg_confidence = sum(r.score for r in results) / len(results)
        
        return Response(
            answer=f"Found {len(results)} relevant passages",
            sources=sources,
            confidence=avg_confidence
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: int) -> Dict:
    """Check document processing status"""
    try:
        status = await db.get_document_status(document_id)
        return {"status": status}
    except Exception as e:
        logger.error(f"Status check failed for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int) -> Dict:
    """Delete a document and its associated data"""
    try:
        await db.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )