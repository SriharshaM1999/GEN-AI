import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "resources/dl-curriculum.pdf")

loader = PyPDFLoader(path)
documents = loader.load()
# print(documents[0].metadata)
print(documents)


"""
Notes:
-----
Requirements:
- run pip install pypdf, inorder to run this file.

"""