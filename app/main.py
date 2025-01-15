
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .database import Database
from .embeddings import Embeddings
from .models import Query, Response, Document
from .utils import load_documents, chunk_text
import os

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(host=os.getenv("QDRANT_HOST", "localhost"))
embeddings = Embeddings(settings.model_name)

@app.post("/documents/upload")
async def upload_documents(file: UploadFile = File(...)):
    file_path = f"/data/documents/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    documents = load_documents(file_path)
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(
            doc["content"],
            settings.chunk_size,
            settings.chunk_overlap
        ))
    
    vectors = embeddings.encode(chunks)
    db.add_documents(settings.collection_name, vectors, chunks)
    
    return {"message": "Documents processed successfully"}

@app.post("/query", response_model=Response)
async def query(query: Query):
    query_vector = embeddings.encode_query(query.text)
    
    results = db.search(
        settings.collection_name,
        query_vector,
        query.top_k
    )
    
    sources = [result.payload for result in results]
    
    return Response(
        answer="Based on the retrieved documents...",
        sources=sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)  
