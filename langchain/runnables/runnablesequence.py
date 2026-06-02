from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

#load env
load_dotenv()

# model
model = ChatOpenAI()


#prompt
prompt1 = PromptTemplate(template="What year is it? {input}", input_variables=["input"])

# output parser
strOutputParser = StrOutputParser()

sequentialChain = RunnableSequence(
    prompt1, model, strOutputParser
)

response = sequentialChain.invoke({"input": "What year is it?"})


print(response)