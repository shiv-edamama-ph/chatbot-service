# Chatbot Using all-MiniLM-L6-v2

This repository provides an implementation of a chatbot using the all-MiniLM-L6-v2 model from the Hugging Face Transformers library. The all-MiniLM-L6-v2 model is a lightweight, fast, and efficient model designed for semantic textual similarity and sentence embeddings.

## Overview

The all-MiniLM-L6-v2 model is well-suited for tasks that require understanding the semantic similarity between sentences. It can be used to create a chatbot that can understand and respond to user queries effectively. This model is efficient and suitable for deployment in environments with limited resources.

## Features

- **Lightweight and Fast**: The all-MiniLM-L6-v2 model provides fast inference times and requires less computational power compared to larger models.
- **Semantic Understanding**: It generates high-quality sentence embeddings that can be used for various NLP tasks, including chatbot responses.
- **Scalability**: Suitable for deployment on devices with limited memory and processing power.

## Installation

To get started, you need to set up your environment and install the necessary dependencies.

### Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate

### To Run a Virtual Environment
uvicorn app.main:app --reload