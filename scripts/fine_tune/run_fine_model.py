import torch
from sentence_transformers import SentenceTransformer, InputExample, SentenceTransformer, util
import pandas as pd

# Load the fine-tuned model
model_path = 'fine-tuned-ecommerce-model'
model = SentenceTransformer(model_path)

# Load your dataset (the same dataset used for fine-tuning)
df = pd.read_csv('data/ecommerce_data.csv')

# Prepare the data for similarity search
responses = df['response'].tolist()

# Function to get the most similar response to a query
def get_response(query, model, responses):
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Encode all responses
    response_embeddings = model.encode(responses, convert_to_tensor=True)
    
    # Compute similarity scores
    similarities = util.pytorch_cos_sim(query_embedding, response_embeddings)
    
    # Get the index of the most similar response
    most_similar_idx = torch.argmax(similarities).item()
    
    return responses[most_similar_idx]

# Example usage
query = "What are the shipping options?"
response = get_response(query, model, responses)
print("Query:", query)
print("Response:", response)
