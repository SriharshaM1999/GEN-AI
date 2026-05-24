from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load env variables from .env file
load_dotenv()

# creating an instance of the OpenAIEmbeddings class with the specified model
#dimension is the number of dimensions in the embedding vector.
embeddingModel = OpenAIEmbeddings(model = "text-embedding-3-small", dimensions=32)

texts = [
    "Kohli is the best batsman in the world",
    "Rohit sharma is a good batsman",
    "Dhoni is a great captain",
    "Sachin is the god of cricket",
    "Virat is a good fielder",
]

# Generate embeddings for the texts
embeddings = embeddingModel.embed_documents(texts)

query = "Who is kholi?"

# Generate embedding for the query
query_embedding = embeddingModel.embed_query(query)

# Calculate cosine similarity between the query embedding and the document embeddings
similarities = cosine_similarity([query_embedding], embeddings)

# Get the index of the most similar document
most_similar_index = np.argmax(similarities)

print(f"Query: {query}")
print(f"Most similar document: {texts[most_similar_index]}")
print(f"Cosine similarity: {similarities[0][most_similar_index]}")