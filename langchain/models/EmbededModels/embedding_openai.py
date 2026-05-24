from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load env variables from .env file
load_dotenv()

# creating an instance of the OpenAIEmbeddings class with the specified model
#dimension is the number of dimensions in the embedding vector.
embeddingModel = OpenAIEmbeddings(model = "text-embedding-3-small", dimensions=32)
# response = embeddingModel.embed_query("What is the capital of India?")


# if you want to generate embeddings of multiple texts, you can use the embed_documents method instead of embed_query.

texts = ["What is the capital of India?", "What is the capital of France?"]
response = embeddingModel.embed_documents(texts)


print(str(response))

"""
Point 1:
---------
response: [0.199462890625, 0.0305633544921875, 0.25537109375, 0.2445068359375, -0.28125, 0.0423583984375, 0.1302490234375, 0.18994140625, 0.084716796875, -0.1297607421875, 0.2091064453125, 0.0469970703125, -0.2205810546875, -0.0272216796875, 0.28466796875, -0.12841796875, -0.109375, 0.1083984375, 0.045013427734375, 0.09051513671875, 0.16064453125, -0.02447509765625, -0.1524658203125, 0.2017822265625,
 0.356689453125, 0.1929931640625, -0.388671875, -0.036590576171875, 0.193603515625, -0.05804443359375, 0.024383544921875, -0.0225067138671875]


 Point 2:
---------
The size of the embedding vector for a small model is 1024 dimensions,
 which means that the embedding vector will have 1024 values. Each value in the embedding vector represents a specific feature or aspect of the input text, and the combination of these values captures the semantic meaning of the text. The size of the embedding vector can vary depending on the model used, with larger models typically producing higher-dimensional embeddings that can capture more complex relationships
 between words and phrases.

 Point 3:
---------
If you want to generate embeddings for multiple texts, you can use the embed_documents method instead of embed_query. The embed_documents method takes a list of texts as input and returns a list of embedding vectors, one for each text. This allows you to efficiently generate embeddings for a batch of texts in a single call to the model, which can be more efficient than generating embeddings for each text individually.
Sample response : 
[[0.199462890625, 0.0305633544921875, 0.25537109375, 0.2445068359375, -0.28125, 0.0423583984375, 0.1302490234375, 0.18994140625, 0.084716796875, -0.1297607421875, 0.2091064453125, 0.0469970703125, -0.2205810546875, -0.0272216796875, 0.28466796875, -0.12841796875, -0.109375, 0.1083984375, 0.045013427734375, 0.09051513671875, 0.16064453125, -0.02447509765625, -0.1524658203125, 0.2017822265625, 0.356689453125, 0.1929931640625, -0.388671875, -0.036590576171875, 0.193603515625, -0.05804443359375, 0.024383544921875, -0.0225067138671875],
 [0.2484130859375, 0.093994140625, 0.1676025390625, 0.1448974609375, -0.1376953125, -0.0164947509765625, -0.08489990234375, 0.0855712890625, 0.0648193359375, -0.0609130859375, 0.041259765625, -0.1436767578125, -0.3671875, -0.08990478515625, -0.0845947265625, 0.13818359375, -0.0394287109375, 0.11578369140625, 0.431884765625, -0.1456298828125, 0.018280029296875, -0.060272216796875, -0.244384765625, 0.0712890625, 0.3701171875, 0.04254150390625, -0.27197265625, -0.043548583984375, 0.0213165283203125, 0.235107421875, 0.251220703125, -0.150146484375]]

"""