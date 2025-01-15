
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import csv
from typing import List, Dict

def load_documents(file_path: str) -> List[Dict]:
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        documents = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                documents.append(row)
        return documents
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return [{"content": f.read()}]
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
