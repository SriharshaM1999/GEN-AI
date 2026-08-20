import streamlit as st
from chatbot import chatbot, get_all_threads
from langchain_core.messages import HumanMessage
import uuid;

#################### utility messages:::::::::


def generate_random_id():  # Step 1.)  generating the new thread-id
    return str(uuid.uuid4());   # str, not UUID: thread ids are compared and rendered as strings

def start_new_thread():
    st.session_state['thread_id'] = generate_random_id();
    if st.session_state['thread_id'] not in st.session_state['all_threads']:
        st.session_state['all_threads'].append(st.session_state['thread_id'])

def clear_messages():
    st.session_state['message_history'] = [];


def load_a_thread_conversation(thread_id: str):
    CONFIG = {"configurable": {"thread_id": thread_id}} # Step 4.) adding the dynamic thread_id.
    # get_state returns a StateSnapshot; the conversation lives in .values['messages'].
    # A thread that was never run has values == {}, hence the .get default.
    snapshot = chatbot.get_state(CONFIG)
    thread_messages = snapshot.values.get('messages', [])

    temp_messages = [];

    for thread_message in thread_messages:
        role = 'user' if isinstance(thread_message, HumanMessage) else 'assistant'
        temp_messages.append({'role': role, 'content': thread_message.content})

    st.session_state['thread_id'] = thread_id;   # resume this thread for the next message
    st.session_state['message_history'] = temp_messages;
        


# """
# sample messages: 
# {'role': user,  'content': 'hi'}
# """


####################### INITIALIZING THE SESSION_VARIABLE to handle state loss on refresh..............!

# streamlit gets refreshed, on user input hence loading the messages array with the data stored in session_state.
if 'message_history' not in st.session_state: # message history in the current thread.
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:  # storing the active/selected thread
    st.session_state['thread_id'] = generate_random_id();


if 'all_threads' not in st.session_state: # preserving all threads that were created.
    st.session_state['all_threads'] = get_all_threads()

if st.session_state['thread_id'] not in st.session_state['all_threads']:    # add the current active thread to all_threads, if not exists.
    st.session_state['all_threads'].append(st.session_state['thread_id']);


############################## SIDE NAV UI

st.sidebar.title("Sriharsha's ChatBot")
if st.sidebar.button('Start a new conversation'):    # on click of start new conversation, create new threa and clear existing messages
    start_new_thread()
    clear_messages()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['all_threads'][::-1]:
    # key= gives each button a stable id of our choosing, instead of one hashed from the label
    if st.sidebar.button(str(thread_id), key=f'thread-{thread_id}'):       # on click on a thread on the left side nav, load the necessary conversationssss.
        load_a_thread_conversation(thread_id)


# ---------------------------


############################## RIGHT PANEL, CHAT PAGE STARTS HERE

# displaying the old messages of the thread
for message in st.session_state['message_history']:       
    with st.chat_message(message['role']):
        st.text(message['content'])


# input for user to type
user_input = st.chat_input('Type here')

# configuration.
CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}} # Step 4.) adding the dynamic thread_id.

# for langsmith, to group the traces of the same thread replace above CONFIG with below

CONFIG = {
    "configurable": {"thread_id": st.session_state['thread_id']},
    "metadata": { # This is important to be passed
        "thread_id": st.session_state["thread_id"],
    },
    "run_name": "chat_turn"
 } # Step 4.) adding the dynamic thread_id.


# on user input
if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        # for message streaming you ahve to use .stream() instead of .invoke() + you have to pass stream_mode argument
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream({'messages':[HumanMessage(content=user_input)]}, CONFIG, stream_mode='messages')
        )
        st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})



