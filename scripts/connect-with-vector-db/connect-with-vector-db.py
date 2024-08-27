#pip install pymongo


# Summary
# Setup MongoDB connection to interact with your database.
# Store embeddings in MongoDB if not already done.
# Perform similarity search using cosine similarity between query and stored embeddings.
# Integrate search functionality into your chatbot backend.

from pymongo import MongoClient
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Adjust the URI as needed
db = client['your_database']
collection = db['your_collection']


# Load the fine-tuned model
model = SentenceTransformer('fine-tuned-ecommerce-model')

# Load your dataset
df = pd.read_csv('ecommerce_data.csv')

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

# Prepare and store embeddings
for _, row in df.iterrows():
    query = row['query']
    response = row['response']
    embedding = model.encode(query).tolist()  # Convert to list for MongoDB storage
    collection.insert_one({
        'query': query,
        'response': response,
        'embedding': embedding
    })



# Load the fine-tuned model
model = SentenceTransformer('fine-tuned-ecommerce-model')

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

def search(query):
    # Generate the query embedding
    query_embedding = model.encode(query).tolist()
    
    # Retrieve all documents from MongoDB
    cursor = collection.find()
    
    # Initialize variables to find the most similar document
    most_similar_doc = None
    highest_similarity = -1
    
    for doc in cursor:
        # Retrieve the stored embedding
        stored_embedding = np.array(doc['embedding'])
        
        # Calculate cosine similarity
        similarity = 1 - cosine(query_embedding, stored_embedding)
        
        # Update the most similar document if needed
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_doc = doc

    return most_similar_doc['response'] if most_similar_doc else "No similar document found."

# Example usage
query = "Tell me about shipping methods."
result = search(query)
print("Most similar response:", result)


print("Embeddings stored in MongoDB.")

