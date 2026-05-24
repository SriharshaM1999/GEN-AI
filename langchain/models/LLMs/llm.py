from langchain_openai import OpenAI
from dotenv import load_dotenv

#loading .env
load_dotenv()

# creating an llm instance
llm = OpenAI(model = "gpt-3.5-turbo-instruct")

inp = input("Query? ")


#invoking the model with the query
response = llm.invoke(input=inp)

print(response)


"""
Point 1:
---------
# This is outdatated code. Langchain has reduced supporting llms, instead it is focusing on supporting 
# chat models. So, we will be using ChatOpenAI instead of OpenAI.


Point 2:
---------
As you can see, the input and output are both strings. This is the standard way of working with LLMs,
     where you provide a string input and get a string output.

"""
