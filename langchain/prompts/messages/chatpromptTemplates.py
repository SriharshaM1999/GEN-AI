from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# loading .env
load_dotenv()


# llm instance
llm = ChatOpenAI(model = "gpt-4", temperature=0.7, max_completion_tokens=300)


#chat prompt template with placeholders for paper_input, style_input and length_input, which will be filled dynamically based on user input in the UI.
chatPromptTemplate = ChatPromptTemplate([
    # if we create system, human messages using below 2 lines, then replacement with user values wont happen
    # SystemMessage(content="You are a {domain} expert"),
    # HumanMessage(content="Please provide a {style_input} explanation of the research paper '{paper_input}' in {length_input}.")

    # follow the below format instead
    ('system', "You are a {domain} expert"),
    # all the old messages will be passed as a list to the placeholder variable.
    MessagesPlaceholder(variable_name="messages"),
    ('human', "Please provide a {style_input} explanation of the research paper '{paper_input}' in {length_input}.")
])

oldMessages = [
    SystemMessage(content="You are a machine learning expert"),
    HumanMessage(content="Please provide a beginner-friendly explanation of the research paper 'Attention Is All You Need'.")
]

prompt = chatPromptTemplate.invoke({
    "domain": "machine learning",
    "style_input": "beginner-friendly",
    "paper_input": "Attention Is All You Need",
    "length_input": "short (1-2 paragraphs)",
    "messages": oldMessages
})

print(prompt)

response = llm.invoke(prompt)
print(response.content)


"""
Notes: 
-------
1.) ChatPromptTemplate allows us to create a prompt template with multiple messages (system, human, ai) 
and placeholders for dynamic values that can be filled based on user input.

2.) It is similar to PromptTemplate but for multiple messages instead of a single prompt string.

"""