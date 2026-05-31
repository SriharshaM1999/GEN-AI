from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import streamlit as st

# loading .env
load_dotenv()

# creating an llm instance
llm = ChatOpenAI(model = "gpt-4", temperature=0.7, max_completion_tokens=1000)

st.title("Sriharsha's GenAI App")
query = st.text_input("Enter your query ?")


if st.button("Submit"):
    response = llm.invoke(input=query)
    st.write(response.content)





# """
# Notes: 

# Dynamic Prompting:
# -----------------
# Dynamic prompting is a technique where the prompt is generated or modified dynamically based on user input, context
# or other factors. This allows for more flexible and personalized interactions with the language model.
# """




