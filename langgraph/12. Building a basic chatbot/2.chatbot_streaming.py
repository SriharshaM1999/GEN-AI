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

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream({'messages':[HumanMessage(content=user_input)]}, CONFIG, stream_mode='messages')
        )
        st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})



