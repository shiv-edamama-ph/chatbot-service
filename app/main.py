from fastapi import FastAPI, HTTPException
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
try:
    index = faiss.read_index('data/index.faiss')
    with open('data/documents.pkl', 'rb') as f:
        documents = pickle.load(f)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Index or documents not found")

@app.post("/chat")
async def chat(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate embedding for query
    query_embedding = model.encode([query])
    
    # Retrieve top matching question
    try:
        D, I = index.search(np.array(query_embedding), 1)
        top_answer = documents[I[0][0]]['answer']
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving answer")

    return {"answer": top_answer}