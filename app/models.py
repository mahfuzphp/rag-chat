
from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    content: str
    metadata: Optional[dict] = None

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 5

class Response(BaseModel):
    answer: str
    sources: List[str]