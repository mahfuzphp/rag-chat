
# File: app/embeddings.py
# Purpose: Text embedding generation
from sentence_transformers import SentenceTransformer
from typing import List

class Embeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str]):
        return self.model.encode(texts)
    
    def encode_query(self, query: str):
        return self.model.encode([query])[0]