from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

#load env
load_dotenv()

# model
model = ChatOpenAI()


#prompt
prompt1 = PromptTemplate(template="write a small tweet in twitter, about {input}", input_variables=["input"])
prompt2 = PromptTemplate(template="write a small tweet in twitter, about {input}", input_variables=["input"])

# output parser
strOutputParser = StrOutputParser()

sequentialChain = RunnableSequence(
    prompt1, model, strOutputParser
)

sequentialChain2 = RunnableSequence(
    prompt2, model, strOutputParser
)

parallelChain = RunnableParallel(
    {"linkedIn": sequentialChain, "twitter": sequentialChain2}
)

response = parallelChain.invoke({"input": "What is the best way to learn programming?"})


print(response)