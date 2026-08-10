from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import BaseMessage, add_messages

load_dotenv()

llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=300, temperature=0.1)


checkpointer = InMemorySaver()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_dessages is a a reducter

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# configuration = {"configurable": {"thread_id": 1}}

# response = chatbot.invoke({"messages": "hi "}, configuration)
# response = chatbot.invoke({"messages": "who is this "}, configuration)

# # you have to pass the configuration to know the history, within in the configuration
# # you can pass checkpointId, to see the history from a specific checkpoint.
# history = chatbot.get_state_history(configuration);

# for record in history: 
#     print(record)



"""

Question : Can I use operator.add instead of add_messages:

Ans:

Yes — Annotated[list[BaseMessage], operator.add] is valid and will append. Any binary callable (existing, update) -> new works as a reducer; operator.add on two lists is just concatenation.

But for messages specifically you lose the things add_messages adds on top of concatenation:

Action         operator.add	    add_messages
Appends lists	✅	              ✅
Coerces {"role": "user", "content": "hi"} / raw strings → HumanMessage	❌ appends the dict as-is	✅
Auto-assigns id to new messages	❌	✅
Upsert by id (re-emit same id → replaces, not duplicates)	❌ duplicates	✅
RemoveMessage / REMOVE_ALL_MESSAGES to trim history	❌ appends a junk object	✅
Rejects a non-list update	💥 TypeError on "messages": response	✅ wraps a single message
So operator.add is the right pick for plain accumulator channels:


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: Annotated[list[str], operator.add]      # log of visited nodes
    scores: Annotated[list[float], operator.add]   # fan-in from parallel nodes
The moment you want to edit, dedupe, or trim conversation history — which you will, since InMemorySaver means the list grows every turn — add_messages is what makes that possible.





"""