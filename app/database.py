
# File: app/database.py
# Purpose: Database connection and operations
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class Database:
    def __init__(self, host: str = "localhost"):
        self.client = QdrantClient(host=host)
        
    def init_collection(self, collection_name: str, vector_size: int):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    
    def add_documents(self, collection_name: str, vectors, documents):
        self.client.upsert(
            collection_name=collection_name,
            points=vectors,
            documents=documents
        )
    
    def search(self, collection_name: str, query_vector, limit: int = 5):
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )