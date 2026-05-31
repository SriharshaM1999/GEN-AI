from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# if we dont have structured output, then we cant chain the reponse from the first model with template2.
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

"""
Notes:
-----
- We have two prompts, one for detailed report and another for summary.
- We use the StrOutputParser to convert the output of the first model into a string that can be used as input for the second prompt.
- The chain allows us to connect the two prompts and the model in between, so that the output of the first prompt can be used as input for the second prompt.
"""
