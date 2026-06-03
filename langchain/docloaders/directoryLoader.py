import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "resources")

loader = DirectoryLoader(path = path, glob = "**/*.pdf", loader_cls=PyPDFLoader)

# LOAD
# documents = loader.load()
# print(documents[0].metadata)
# print(documents)

# LAZY LOAD
for document in loader.lazy_load():
    print(document.metadata)
    print(document.page_content[:100])

"""
Notes: 

- The DirectoryLoader class is used to load documents from a specified directory path.
- The glob parameter allows you to specify a pattern to match files in the directory. In this case, it will load all PDF files in the directory and its subdirectories.
- The loader_cls parameter allows you to specify the loader class to use for loading the documents. In this case, we are using the PyPDFLoader to load PDF files.
- The load() method reads the content of the files and returns it as a List<Document>
- Each document is represented as a dictionary with the following keys:
  - 'page_content': The actual text content of the document.
  - 'metadata': A dictionary containing metadata about the document, such as the source file path


----
- The DirectoryLoader loads all the files at once, which can be memory intensive if there are many large files. In such cases, consider using a streaming approach or loading files in batches.
- So it is suggested to use lazy_load() method instead of load() method, which will load the documents one by one, instead of loading all the documents at once.
"""