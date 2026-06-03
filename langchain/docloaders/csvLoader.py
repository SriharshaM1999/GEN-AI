from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1]) # row 1 of the csv file

"""
Notes:
-----
- The CSVLoader class is used to load CSV files from a specified file path.
- The load() method reads the content of the CSV file and returns it as a List<Document>
- Each document is represented as a dictionary with the following
    - 'page_content': The actual text content of the document.
    - 'metadata': A dictionary containing metadata about the document, such as the source file path
"""