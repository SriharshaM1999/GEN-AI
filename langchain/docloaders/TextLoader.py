import os
from langchain_community.document_loaders import TextLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "resources/cricket.txt")

loader = TextLoader(path)
documents = loader.load()
print(documents)

"""
Notes:
-----
- The TextLoader class is used to load text documents from a specified file path.
- The load() method reads the content of the file and returns it as a List<Document>
- Each document is represented as a dictionary with the following keys:
  - 'page_content': The actual text content of the document.
  - 'metadata': A dictionary containing metadata about the document, such as the source file path
"""