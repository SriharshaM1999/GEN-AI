from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/?envType=daily-question&envId=2026-03-07'
loader = WebBaseLoader(url) # we can also pass a list of urls to load multiple documents at once

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'solution to this problem?', 'text':docs[0].page_content}))

"""
Notes:
------
- The WebBaseLoader class is used to load web pages from a specified URL or a list of URLs.
- The load() method reads the content of the web page(s) and returns it as a List<Document>
- Each document is represented as a dictionary with the following keys:
  - 'page_content': The actual text content of the web page.
  - 'metadata': A dictionary containing metadata about the web page, such as the source URL

  
----
- we can also pass a list of urls to load multiple documents at once, and the load() method will return a list of documents corresponding to each url.
"""