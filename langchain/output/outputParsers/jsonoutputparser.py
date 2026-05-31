from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    # this is a partial_variable because, this variable is not going to be provided by the user, value will not be inserted at runtime.
    partial_variables={'format_instruction': parser.get_format_instructions()} 
)

chain = template | model | parser

# commenting out the actual invocation to avoid hitting the API during testing
# result = chain.invoke({'topic':'black hole'})

# print(result)

"""
Notes:
-----
- We are using the HuggingFaceEndpoint to connect to a Hugging Face model.
- The JsonOutputParser is used to parse the output of the model into a JSON format.
- The format instruction is provided to the model as a partial variable, which means that it will not be provided by the user at runtime,
 but it will be included in the prompt when it is sent to the model. This allows us to instruct the model on how to format its output so 
 that it can be correctly parsed by the JsonOutputParser.

 -----
 - ** This JsonOutputParser, wont give you feasibility to add your own custom keys in the output.
"""
