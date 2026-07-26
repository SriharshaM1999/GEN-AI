from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

path = os.path.dirname(__file__)

loader = PyPDFLoader('' + path + '/../docloaders/resources/dl-curriculum.pdf')

# loading the document.
docs = loader.load()

# splitter to split the document into chunks of 200 characters with no overlap and no separator.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

# to load documents and split them into chunks.
result = splitter.split_documents(docs)

# print(result[1].page_content) # the result is a list of documents, where each document is a chunk of the original document.



text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

# to split the text into chunks of 200 characters with no overlap and no separator.
result = splitter.split_text(text)

print(result[1])


"""
Notes:
-----
TextSplitter is a class that can be used to split both document object and text into smaller chunks.
The CharacterTextSplitter is a subclass of TextSplitter that splits the document/text into chunks of a specified number of characters.
The chunk_overlap parameter specifies the number of characters that should overlap between chunks,
 and the separator parameter specifies the string that should be used to separate the chunks.

"""