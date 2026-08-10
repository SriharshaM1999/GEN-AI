import streamlit as st
from chatbot import chatbot
from langchain_core.messages import HumanMessage

# """
# sample messages: 
# {'role': user,  'content': 'hi'}
# """

# streamlit gets refreshed, on user input hence loading the messages array with the data stored in session_state.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# displaying the old messages of the thread
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# input for user to type
user_input = st.chat_input('Type here')

# configuration.
CONFIG = {"configurable": {"thread_id": 'thread-1'}}

# on user input
if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    reponse = chatbot.invoke({'messages':[HumanMessage(content=user_input)]}, CONFIG)
    last_message = reponse['messages'][-1].content
    print("last_message", last_message)


    for record in chatbot.get_state_history(CONFIG):
        print(record)

    st.session_state['message_history'].append({'role':'assistant', 'content': last_message})
    with st.chat_message('assistant'):
        st.text(last_message)




