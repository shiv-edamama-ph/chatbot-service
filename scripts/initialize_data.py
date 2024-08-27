#!/usr/bin/env python

import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Paths
dataset_path = 'data/train.json'
index_path = 'data/index.faiss'
embeddings_path = 'data/embeddings.npy'
documents_path = 'data/documents.pkl'

# Check if dataset file exists
if not os.path.isfile(dataset_path):
    print(f"Error: {dataset_path} not found!")
    exit(1)

print("Dataset found. Creating embeddings and FAISS index...")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
with open(dataset_path, 'r') as f:
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
print("Saving embeddings and FAISS index...")
faiss.write_index(index, index_path)
np.save(embeddings_path, embeddings)

# Save documents
print("Saving documents...")
with open(documents_path, 'wb') as f:
    pickle.dump(data, f)

print("Embeddings, FAISS index, and documents saved successfully.")
