# Main FastAPI application file implementing a RAG (Retrieval-Augmented Generation) system
# This file handles document uploads, querying, and system health monitoring

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

# Import our application components
from config import settings
from database import Database
from models import Query, Response, Document, SearchResult
from embeddings import Embeddings
from text_processor import TextProcessor
from utils import setup_logging

# Initialize logging
logger = setup_logging()

# Create FastAPI application
app = FastAPI(
    title="RAG System API",
    description="A Retrieval-Augmented Generation system for document processing and querying",
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

# Initialize components
db = Database()
embeddings = Embeddings(settings.model_name)
text_processor = TextProcessor(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    logger.info("Starting RAG system...")
    try:
        # Initialize database connections
        await db.initialize()
        logger.info("Database connections established")
        
        # Ensure required directories exist
        os.makedirs(settings.documents_dir, exist_ok=True)
        logger.info("Document directory initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down RAG system...")
    await db.close_connections()

@app.get("/health")
async def health_check() -> Dict:
    """
    Check system health and component status
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
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
    """
    Upload and process documents for the RAG system
    The function handles document chunking, embedding generation, and storage
    """
    try:
        logger.info(f"Processing upload: {file.filename}")
        
        # Read and validate file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Store original document metadata
        doc_id = await db.store_document_metadata({
            "filename": file.filename,
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
            "content_type": file.content_type
        })
        
        # Process document in background if requested
        async def process_document():
            try:
                # Convert content to text and chunk it
                text_content = content.decode('utf-8')
                chunks = text_processor.chunk_text(text_content)
                
                # Generate embeddings for chunks
                chunk_embeddings = embeddings.encode_batch(chunks)
                
                # Store chunks and their embeddings
                await db.store_chunks(doc_id, chunks, chunk_embeddings)
                
                logger.info(f"Successfully processed document {file.filename}")
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
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query) -> Response:
    """
    Query the RAG system to find relevant document chunks
    The function generates query embeddings and performs semantic search
    """
    try:
        logger.info(f"Processing query: {query.text[:100]}...")
        
        # Generate embedding for query
        query_embedding = embeddings.encode_text(query.text)
        
        # Search for relevant chunks
        results = await db.semantic_search(
            query_embedding,
            limit=query.top_k or 5,
            threshold=query.threshold or 0.7
        )
        
        if not results:
            logger.warning("No relevant results found")
            return Response(
                answer="No relevant documents found",
                sources=[],
                confidence=0.0
            )
        
        # Format results
        sources = [
            SearchResult(
                text=result.text,
                confidence=result.score,
                document_id=result.document_id,
                metadata=result.metadata
            )
            for result in results
        ]
        
        # Calculate overall confidence
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
    """
    Check the processing status of a specific document
    """
    try:
        status = await db.get_document_status(document_id)
        return {"status": status}
    except Exception as e:
        logger.error(f"Status check failed for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int) -> Dict:
    """
    Delete a document and its associated chunks from the system
    """
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