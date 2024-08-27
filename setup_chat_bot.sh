#!/bin/bash

# Set up directories
echo "Creating project directories..."
mkdir -p rag_chatbot/{app/{routes,services,utils},data,scripts}
cd rag_chatbot || { echo "Failed to change directory"; exit 1; }

# Set up Python virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Install necessary Python packages
echo "Installing Python dependencies..."
pip install fastapi uvicorn sentence-transformers faiss-cpu transformers datasets || { echo "Failed to install dependencies"; exit 1; }

# Create dataset placeholder
echo "Creating a sample dataset..."
cat <<EOT > data/train.json
[
  {
    "question": "How can I track my order?",
    "answer": "You can track your order through the 'Track Order' section on our website."
  },
  {
    "question": "What is your return policy?",
    "answer": "Our return policy allows returns within 30 days of purchase."
  }
]
EOT

# Create embeddings and FAISS index script
echo "Creating script to generate embeddings and FAISS index..."
cat <<EOT > scripts/create_embeddings.py
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

# Load dataset
with open('data/train.json', 'r') as f:
    data = json.load(f)

questions = [entry['question'] for entry in data]

# Create embeddings
print("Generating embeddings...")
embeddings = model.encode(questions)

# Create FAISS index
print("Creating FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save the index and embeddings
faiss.write_index(index, 'data/index.faiss')
np.save('data/embeddings.npy', embeddings)

# Save documents
with open('data/documents.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Embeddings and FAISS index created successfully.")
EOT

# Run the script to create embeddings and FAISS index
echo "Generating embeddings and FAISS index..."
python scripts/create_embeddings.py || { echo "Failed to run create_embeddings.py"; exit 1; }

# Create FastAPI main application
echo "Creating FastAPI application..."
cat <<EOT > app/main.py
from fastapi import FastAPI
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('data/index.faiss')
with open('data/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

@app.post("/chat")
async def chat(query: str):
    # Generate embedding for query
    query_embedding = model.encode([query])
    
    # Retrieve top matching question
    D, I = index.search(np.array(query_embedding), 1)
    
    # Get the most relevant answer
    top_answer = documents[I[0][0]]['answer']
    
    return {"answer": top_answer}
EOT

# Create a route definition
echo "from fastapi import APIRouter" > app/routes/__init__.py

echo "Setup completed successfully!"
echo "You can now run the FastAPI application with the following command:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload"
