# LangGraph (+ LangSmith, MCP): quick revision

Source folders: [langgraph/](../langgraph/), [langsmith/](../langsmith/), [mcp/](../mcp/)

Common imports used by the snippets:
```python
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

---

## 1. Generative AI vs Agentic AI
- **Generative AI** responds to a prompt and makes content (text, images, code). It's reactive.
- **Agentic AI** is given a **goal** and works toward it on its own: it plans, acts with tools, watches the results, and adapts. A human steps in only when needed.
- Traits: autonomy, goal-oriented, planning, reasoning, adaptability, context awareness (memory).
- Parts: **brain** (LLM), **orchestrator** (the control flow), **tools**, **memory**, **supervisor** (human in the loop, guardrails).

## 2. LangChain vs LangGraph
- LangChain chains are linear (DAG-like). They struggle with:
  - conditional branching
  - **loops** (retry until good)
  - jumping back to an earlier step
  - shared **state**
  - **pausing for a human** (human in the loop)
  - **fault tolerance** (resuming after a crash)
- LangGraph models the app as a **graph**: nodes, edges, a shared state, and checkpointing built in.
- Use LangChain for components (models, prompts, retrievers, tools). Use LangGraph for the orchestration.

## 3. Core concepts
- **LLM workflow patterns**:
  - prompt chaining
  - routing
  - parallelization
  - orchestrator-workers
  - evaluator-optimizer
- **State**: a `TypedDict` (or Pydantic model) shared by every node. Each node gets the state and returns **only the keys it changed**.
- **Nodes**: Python functions. **Edges**: `add_edge(a, b)`, `add_conditional_edges(a, router_fn, {label: node})`. Special nodes: `START` and `END`.
- **Reducers** decide how an update merges into the state. Without one, the new value **replaces** the old.
  - `Annotated[list[int], operator.add]` concatenates lists. Used for parallel scores and the tweet history.
  - `Annotated[list[BaseMessage], add_messages]` appends messages (and updates a message whose id already exists).
- **Execution model** (inspired by Pregel): define the graph → `compile()` (checks the structure) → `invoke(initial_state)`. Nodes run in **super-steps**. Parallel branches run in the same super-step. The run stops when no node is active.
- Draw the graph: `Image(workflow.get_graph().draw_mermaid_png())`, or just leave `workflow` as the last line of a notebook cell.

```python
class State(TypedDict):
    scores: Annotated[list[int], operator.add]           # reducer: lists are concatenated
    messages: Annotated[list[BaseMessage], add_messages]  # reducer: messages are appended
    result: str                                           # no reducer: replaced

def my_node(state: State) -> dict:
    return {"result": "done"}            # return only what changed

graph = StateGraph(State)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)
workflow = graph.compile()
workflow.invoke({"scores": [], "messages": [], "result": ""})

from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())
```

## 4. Workflows
- **Sequential** ([6. SequentialWorkflows/](<../langgraph/6. SequentialWorkflows/>)):
  - BMI: calculate → categorize.
  - LLM question and answer.
  - **Prompt chaining**: outline → blog post.
- **Parallel** ([7. Parallel Workflows/](<../langgraph/7. Parallel Workflows/>)):
  - Fan out: several `add_edge(START, x)`. Fan in: every branch has an edge to `summary`.
  - Cricket stats; UPSC essay graded on 3 aspects with `with_structured_output`, then the scores averaged.
  - **Parallel nodes must return partial dicts.** Returning the whole state gives `InvalidUpdateError`, because two branches write the same key in one step. When two branches really must write one key, give that key a reducer (`operator.add`).
- **Conditional** ([8. ConditionalWorkflows/](<../langgraph/8. ConditionalWorkflows/>)):
  - The router returns the **name of the next node**, typed as `Literal[...]`.
  - Quadratic: discriminant → real, repeated or no roots.
  - Review reply: sentiment → positive reply, or diagnosis (issue, tone, urgency) → negative reply.
- **Iterative** ([9. IterativeWorkflows/](<../langgraph/9. IterativeWorkflows/>)):
  - Tweet: generate → evaluate (structured rating and feedback) → `route` sends it to `END` if approved or **back to generate**. This is evaluator-optimizer.
  - Always cap it with `max_retries` / `retry_count`, or it can loop forever.
  - On retry, pass the feedback to the generator.

**Sequential (prompt chaining)**
```python
class BlogState(TypedDict):
    topic: str
    outline: str
    blog: str

def make_outline(s: BlogState):
    return {"outline": llm.invoke(f"Outline for a blog on {s['topic']}").content}

def make_blog(s: BlogState):
    return {"blog": llm.invoke(f"Write a blog on {s['topic']} using:\n{s['outline']}").content}

g = StateGraph(BlogState)
g.add_node("make_outline", make_outline)
g.add_node("make_blog", make_blog)
g.add_edge(START, "make_outline")
g.add_edge("make_outline", "make_blog")
g.add_edge("make_blog", END)
g.compile().invoke({"topic": "AI in healthcare"})
```

**Parallel (fan out, fan in)**
```python
class Eval(BaseModel):
    feedback: str
    score: int = Field(ge=0, le=10)
judge = llm.with_structured_output(Eval)

class EssayState(TypedDict):
    essay: str
    language_fb: str
    analysis_fb: str
    scores: Annotated[list[int], operator.add]   # both branches write here
    avg: float

def language(s):
    out = judge.invoke(f"Rate the language:\n{s['essay']}")
    return {"language_fb": out.feedback, "scores": [out.score]}     # partial update

def analysis(s):
    out = judge.invoke(f"Rate the analysis:\n{s['essay']}")
    return {"analysis_fb": out.feedback, "scores": [out.score]}

def final(s):
    return {"avg": sum(s["scores"]) / len(s["scores"])}

g = StateGraph(EssayState)
for name, fn in [("language", language), ("analysis", analysis), ("final", final)]:
    g.add_node(name, fn)
g.add_edge(START, "language"); g.add_edge(START, "analysis")   # fan out
g.add_edge("language", "final"); g.add_edge("analysis", "final")  # fan in
g.add_edge("final", END)
```

**Conditional (routing)**
```python
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    reply: str

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"]

def find_sentiment(s):
    return {"sentiment": llm.with_structured_output(Sentiment).invoke(s["review"]).sentiment}

def positive_reply(s): return {"reply": "Thanks a lot!"}
def negative_reply(s): return {"reply": "Sorry, we are fixing it."}

def route(s) -> Literal["positive_reply", "negative_reply"]:
    return "positive_reply" if s["sentiment"] == "positive" else "negative_reply"

g = StateGraph(ReviewState)
g.add_node("find_sentiment", find_sentiment)
g.add_node("positive_reply", positive_reply)
g.add_node("negative_reply", negative_reply)
g.add_edge(START, "find_sentiment")
g.add_conditional_edges("find_sentiment", route)       # router returns the next node's name
g.add_edge("positive_reply", END); g.add_edge("negative_reply", END)
```

**Iterative (evaluator-optimizer loop)**
```python
class Rating(BaseModel):
    rating: int = Field(ge=1, le=5)
    feedback: str

class TweetState(TypedDict):
    topic: str
    tweet: str
    rating: int
    retry: int
    max_retries: int
    feedback: Annotated[list[str], operator.add]

def generate(s):
    fb = s["feedback"][-1] if s["feedback"] else "none"
    tweet = llm.invoke(f"Tweet about {s['topic']}. Previous: {s['tweet']}. Fix: {fb}").content
    return {"tweet": tweet, "retry": s["retry"] + 1}

def evaluate(s):
    out = llm.with_structured_output(Rating).invoke(f"Rate 1-5 with feedback: {s['tweet']}")
    return {"rating": out.rating, "feedback": [out.feedback]}

def route(s) -> Literal["approved", "retry"]:
    return "approved" if s["rating"] >= 4 or s["retry"] >= s["max_retries"] else "retry"

g = StateGraph(TweetState)
g.add_node("generate", generate)
g.add_node("evaluate", evaluate)
g.add_edge(START, "generate")
g.add_edge("generate", "evaluate")
g.add_conditional_edges("evaluate", route, {"approved": END, "retry": "generate"})
g.compile().invoke({"topic": "AI", "tweet": "", "rating": 0, "retry": 0, "max_retries": 3, "feedback": []})
```

## 5. Chatbot + persistence ([10. Chatbot/](<../langgraph/10. Chatbot/>), [11. PersistantStorage/](<../langgraph/11. PersistantStorage/>))
- State: `messages: Annotated[list[BaseMessage], add_messages]`. The node returns `{"messages": [response]}`.
- Without a checkpointer, every `invoke` starts from nothing and nothing is remembered.
- **Checkpointer** saves the state after **every super-step**, keyed by `thread_id`:
  - `compile(checkpointer=InMemorySaver())`
  - `invoke(state, {"configurable": {"thread_id": "1"}})`
- What persistence gives you:
  - **Short-term memory**: the chat history for each thread.
  - `get_state(config)`: the latest `StateSnapshot`. Use `.values` for the data and `.next` for the node that runs next.
  - `get_state_history(config)`: every checkpoint, newest first.
  - **Time travel**: put `checkpoint_id` in the config, then `invoke(None, config)` replays from that point.
  - **Edit the state**: `update_state(config_with_checkpoint_id, {"topic": "banana"})` creates a new checkpoint, which you can then resume.
  - **Fault tolerance**: if a node crashes, `invoke(None, config)` resumes from the last saved checkpoint instead of starting over.
  - **Human in the loop**: a run can pause and resume later.
- In production: `SqliteSaver(sqlite3.connect("chatbot.db", check_same_thread=False))` (`langgraph-checkpoint-sqlite`), or `PostgresSaver`.
- List every thread: `{c.config["configurable"]["thread_id"] for c in checkpointer.list(None)}`.

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    return {"messages": [llm.invoke(state["messages"])]}

g = StateGraph(ChatState)
g.add_node("chat_node", chat_node)
g.add_edge(START, "chat_node"); g.add_edge("chat_node", END)
chatbot = g.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}
chatbot.invoke({"messages": [HumanMessage("I am Harsha")]}, config)
chatbot.invoke({"messages": [HumanMessage("What is my name?")]}, config)   # remembers

snap = chatbot.get_state(config)
snap.values["messages"], snap.next
history = list(chatbot.get_state_history(config))                        # newest first

# time travel: replay from an old checkpoint
old = {"configurable": {"thread_id": "1", "checkpoint_id": history[-2].config["configurable"]["checkpoint_id"]}}
chatbot.invoke(None, old)

# edit the state at that checkpoint, then resume from the new checkpoint
new_config = chatbot.update_state(old, {"messages": [HumanMessage("Actually I am Ravi")]})
chatbot.invoke(None, new_config)

# fault tolerance: after a crash in a node, resume the same thread
chatbot.invoke(None, config)

# persistent storage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver(sqlite3.connect("chatbot.db", check_same_thread=False))
chatbot = g.compile(checkpointer=checkpointer)
threads = {c.config["configurable"]["thread_id"] for c in checkpointer.list(None)}
```

## 6. Streamlit chatbot UI ([12. Building a basic chatbot/](<../langgraph/12. Building a basic chatbot/>))
- Streamlit **re-runs the whole script on every input**, so keep the UI history in `st.session_state`.
- `st.chat_input`, `st.chat_message(role)`.
- **Streaming** uses `stream_mode="messages"` with `st.write_stream`.
- **Multiple threads**:
  - Make a new `uuid4()` per chat.
  - A sidebar lists the threads.
  - Opening a thread loads `chatbot.get_state(config).values["messages"]` and turns them into role/content pairs.
- **Persistent**: switching to `SqliteSaver` keeps the threads across app restarts.

```python
import uuid
import streamlit as st
from chatbot import chatbot          # the compiled graph from section 5

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# sidebar: new chat + old threads
if st.sidebar.button("New chat"):
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message_history"] = []
for tid in get_all_threads():
    if st.sidebar.button(tid):
        msgs = chatbot.get_state({"configurable": {"thread_id": tid}}).values.get("messages", [])
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in msgs]

for m in st.session_state["message_history"]:
    with st.chat_message(m["role"]):
        st.text(m["content"])

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
if user_input := st.chat_input("Type here"):
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)
    with st.chat_message("assistant"):
        reply = st.write_stream(
            chunk.content for chunk, meta in
            chatbot.stream({"messages": [HumanMessage(user_input)]}, CONFIG, stream_mode="messages"))
    st.session_state["message_history"].append({"role": "assistant", "content": reply})
```

## 7. LangSmith ([langsmith/langsmith/](../langsmith/langsmith/))
- Observability for LLM apps: a **trace** for every run, showing inputs and outputs, latency, tokens, cost and errors at each step.
- Setup is just env vars: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`. LangChain and LangGraph runs are then traced automatically.
- In the config: `run_name`, `tags`, `metadata`.
- `@traceable(name=..., tags=...)` traces plain Python functions too (load PDF, split, build index). Nest traced functions to group steps under one parent run.
- RAG v1 → v4 progression: plain chain → traced steps → a parent setup function → cached FAISS index (fingerprint of the file).
- Agents and LangGraph: every tool call and node shows up as a child run.
- Other features: datasets, evaluation (LLM-as-judge), monitoring dashboards, prompt hub, feedback.

```python
# .env
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_...
# LANGCHAIN_PROJECT=my-project

import os
os.environ["LANGCHAIN_PROJECT"] = "Report then summary"    # set before the run

config = {"run_name": "report_chain",
          "tags": ["report", "summary"],
          "metadata": {"model": "gpt-4o-mini"}}
chain.invoke({"topic": "Unemployment in India"}, config)

from langsmith import traceable

@traceable(name="load_pdf")
def load_pdf(path): return PyPDFLoader(path).load()

@traceable(name="setup_pipeline", tags=["setup"])
def setup(path):
    docs = load_pdf(path)               # shows up as a child run
    return FAISS.from_documents(split(docs), OpenAIEmbeddings())

# group a chatbot's traces by thread
chatbot.invoke(state, {"configurable": {"thread_id": tid}, "metadata": {"thread_id": tid}, "run_name": "chat_turn"})
```

## 8. Tools in LangGraph ([13. Tools Integration/](<../langgraph/13. Tools Integration/>))
- The tool node must be named **"tools"** (that's the name `tools_condition` returns).
- Don't also add `chat → END`. `tools_condition` already handles the end.
- `parallel_tool_calls=False` forces one tool call at a time.

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def calculator(a: int, b: int, op: Literal["add", "sub", "mul", "div"]) -> float:
    """Do basic arithmetic on two numbers"""
    return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]

tools = [DuckDuckGoSearchRun(), calculator]
llm_tools = llm.bind_tools(tools)

def chat_node(state: ChatState):
    return {"messages": [llm_tools.invoke(state["messages"])]}

g = StateGraph(ChatState)
g.add_node("chat", chat_node)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "chat")
g.add_conditional_edges("chat", tools_condition)   # tool_calls -> "tools", else END
g.add_edge("tools", "chat")                        # loop back with the ToolMessage
app = g.compile(checkpointer=InMemorySaver())

for update in app.stream({"messages": [HumanMessage("What is 12 * 9?")]},
                         {"configurable": {"thread_id": "1"}}, stream_mode="updates"):
    print(update)
```

## 9. MCP ([14. Langgraph with MCP/notes.md](<../langgraph/14. Langgraph with MCP/notes.md>), [mcp/expense-tracker/](../mcp/expense-tracker/))
- **Model Context Protocol** is a standard way to expose tools, resources and prompts. Write the server once and any client can use it (LangGraph, Claude Desktop, Cursor).
- Server: `FastMCP(name)`, `@mcp.tool`, `mcp.run(transport="stdio")`.
  - With stdio, **log to stderr**, because stdout carries the protocol.
- Client:
  - `MultiServerMCPClient({"math": {"transport": "stdio", "command": sys.executable, "args": [abs_path]}})`
  - `tools = await client.get_tools()`. Call it **once**, outside the nodes.
- MCP tools are async, so use `await workflow.ainvoke(...)`.
- Transports: `stdio` (local subprocess), `streamable_http` / `sse` (remote server at a URL).
- CLI:
  - `fastmcp dev inspector server.py` opens a web UI to test tools by hand.
  - `fastmcp run`, `fastmcp list`, `fastmcp call`.
  - `fastmcp install claude-desktop server.py` registers the server in a client.

**Server** (`server.py`)
```python
import sys
from fastmcp import FastMCP
mcp = FastMCP(name="MathServer")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print("add called", file=sys.stderr)    # never print to stdout with stdio
    return a + b

@mcp.resource("info://version")
def version() -> str:
    return "1.0"

@mcp.prompt
def explain(topic: str) -> str:
    return f"Explain {topic} simply"

if __name__ == "__main__":
    mcp.run(transport="stdio")              # or transport="http", port=8000
```

**Client in LangGraph** (notebook, so top-level `await` works)
```python
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "math": {"transport": "stdio", "command": sys.executable, "args": ["/abs/path/server.py"]},
    # "remote": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
})
tools = await client.get_tools()            # once, at startup
llm_tools = llm.bind_tools(tools)

async def chat_node(state: ChatState):
    return {"messages": [await llm_tools.ainvoke(state["messages"])]}

g = StateGraph(ChatState)
g.add_node("chat", chat_node)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "chat")
g.add_conditional_edges("chat", tools_condition)
g.add_edge("tools", "chat")
app = g.compile()
await app.ainvoke({"messages": [HumanMessage("What is 1 + 2?")]})
```

---

## Self-test
1. What problems does LangGraph solve that a LangChain chain can't?
2. What is a reducer? What happens to a list key in the state without one?
3. Why do parallel nodes return partial dicts? Which error do you get otherwise?
4. What does a conditional edge's router return?
5. What does a checkpointer store, and when? What is `thread_id`?
6. How do you time travel to an old checkpoint and resume? How do you change the state there first?
7. Why must the tool node be called `"tools"`? What does `tools_condition` return?
8. Why does `.invoke()` fail with MCP tools? Why is `get_tools()` inside a node a bad idea?
9. Name the 3 LangSmith env vars, and what `@traceable` adds.
