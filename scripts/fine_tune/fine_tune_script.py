import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader

# Define the relative path to your CSV file
relative_path = 'data/ecommerce_data.csv'

# Construct the full path dynamically based on the script location
base_dir = os.path.dirname(__file__)  # Directory where the script is located
file_path = os.path.join(base_dir, relative_path)

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your dataset
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path and filename.")
    raise  # Re-raise the exception to stop execution
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    raise  # Re-raise the exception to stop execution

# Validate and preprocess dataset
required_columns = ['query', 'response']

# Check if required columns exist in the dataset
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset must contain columns: {required_columns}")

# Drop rows with missing or empty values in required columns
df.dropna(subset=required_columns, inplace=True)
df = df[df[required_columns].apply(lambda x: x.str.strip().astype(bool)).all(axis=1)]

# Check the first few rows to verify correctness
print(df.head())

# Prepare training data
train_samples = []
for _, row in df.iterrows():
    query = str(row['query']).strip()
    response = str(row['response']).strip()
    if query and response:  # Ensure both fields are non-empty
        train_samples.append(InputExample(texts=[query, response]))

# Check if we have any training samples
if not train_samples:
    raise ValueError("No valid training samples found. Please check the dataset.")

# Convert to a DataLoader
train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

# Define a loss function
train_loss = losses.MultipleNegativesRankingLoss(model)
# Fine-tune the model
num_epochs = 1
output_dir = 'fine-tuned-ecommerce-model'

try:
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=100,
              output_path=output_dir)
    print(f"Fine-tuning complete. Model saved to '{output_dir}'")
except Exception as e:
    print(f"An error occurred during fine-tuning: {e}")


# Explanation
# Loading the Model:

# The pre-trained all-MiniLM-L6-v2 model is loaded.
# Loading the Dataset:

# The dataset is loaded from a CSV file into a DataFrame.
# Preparing Training Data:

# Each row from the dataset is converted into InputExample instances.
# Creating DataLoader:

# The dataset is wrapped in a DataLoader for batching.
# Defining the Loss Function:

# CosineSimilarityLoss is used for training.
# Fine-Tuning:

# The model is fine-tuned for a specified number of epochs and saved.

# pip install pandas sentence-transformers
