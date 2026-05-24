from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the captial of India"

response = embedding.embed_query(text)
print(str(response))

texts = ["Delhi is the captial of India", "Paris is the capital of France"]
response = embedding.embed_documents(texts)
print(str(response))


"""
Note 1: This code wont work, as i didn't add huggingface token in the .env file. To make this code work, you need to add your Hugging Face API token to the .env file in the following format:
HUGGINGFACEHUB_API_TOKEN=your_token_here

"""