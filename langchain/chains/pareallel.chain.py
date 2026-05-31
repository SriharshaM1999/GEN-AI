from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

#load .env file
load_dotenv()


prompt1 = PromptTemplate(
    template='Prepare a list of 5 points about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Prepare a list of 5 questions about {topic}',
)

prompt3 = PromptTemplate(
    template = 'Merge the outputs of {notes} and {questions} into a single list of 10 points',
    input_variables=['notes', 'questions']
)

# create a model
model = ChatOpenAI()

# create a output parser
parser = StrOutputParser()

# create a parallel chain
parallelChain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'questions': prompt2 | model | parser
})


finalChain = parallelChain | prompt3 | model | parser

response = finalChain.invoke({'topic':'cricket'})


print(response) # no need to access context property as we have output parsers here.

print(finalChain.get_graph().print_ascii())

"""
Notes:
-----
- This is an example of how to use parallel chain
- Here we used RunnableParallel to create a parallel chain that runs two chains in parallel and then merges their outputs using another chain
    parallelChain = RunnableParallel({
        'notes': prompt1 | model | parser,  # notes chain, so output will be stored in context with key 'notes'
        'questions': prompt2 | model | parser # questions chain, so output will be stored in context with key 'questions'
    })
- The final chain takes the outputs of the parallel chains and merges them using prompt3 and model
"""


