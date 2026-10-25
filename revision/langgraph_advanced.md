# LangGraph: advanced and missing topics

These are topics from the playlist that your notes don't cover yet, mostly from the videos you haven't watched. Read [02_langgraph.md](02_langgraph.md) first.

```python
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o-mini")
```

`MessagesState` is a built-in state with `messages: Annotated[list[BaseMessage], add_messages]`. You can subclass it to add more keys.

---

## 1. Execution model: super-steps, `Send`, `Command`
- LangGraph runs in rounds called **super-steps**, inspired by Google's Pregel:
  1. Every node that has an incoming message is **active**.
  2. The active nodes run, in parallel if there are several.
  3. Their updates are merged into the state **through the reducers**, all at once at the end of the step.
  4. The next nodes get activated. The run stops when no node is active.
- That's why two parallel nodes writing the same key without a reducer raise `InvalidUpdateError`: two updates arrive in one step and LangGraph can't choose between them.
- A checkpoint is saved after **every super-step**. Fault tolerance and time travel work at the level of one step.
- **Loop safety**: `invoke(state, {"recursion_limit": 10})` caps the number of steps and raises `GraphRecursionError` if the cap is hit.

**`Send`: map-reduce, the orchestrator-workers pattern.** Use it when you only know at runtime how many parallel branches you need (one per section, per document, per subtopic).
```python
from langgraph.types import Send

class State(TypedDict):
    topic: str
    sections: list[str]
    drafts: Annotated[list[str], operator.add]

def plan(s):
    return {"sections": ["intro", "history", "future"]}     # in practice an LLM picks these

def fan_out(s):
    return [Send("write", {"section": sec, "topic": s["topic"]}) for sec in s["sections"]]

def write(w):                                              # each worker gets its own small input
    return {"drafts": [llm.invoke(f"Write the {w['section']} of a blog on {w['topic']}").content]}

g = StateGraph(State)
g.add_node("plan", plan)
g.add_node("write", write)
g.add_edge(START, "plan")
g.add_conditional_edges("plan", fan_out, ["write"])
g.add_edge("write", END)
g.compile().invoke({"topic": "AI", "sections": [], "drafts": []})
```

**`Command`: update the state and pick the next node in one return.** It replaces a separate router function.
```python
from langgraph.types import Command

def check(s) -> Command[Literal["approve", "reject"]]:
    ok = s["score"] > 5
    return Command(update={"status": "ok" if ok else "bad"}, goto="approve" if ok else "reject")
```

## 2. Human in the loop (HITL)
- Use it before risky actions (sending email, payments, deleting data), or when the agent needs facts only a person has.
- `interrupt(payload)` inside a node **pauses** the graph and saves the state. **A checkpointer is required.**
- `invoke` returns early, and `result["__interrupt__"]` holds the payload. Show it to the user.
- Resume with `invoke(Command(resume=value), same_config)`. `interrupt()` then **returns that value**.
- Gotcha: on resume, the **node runs again from its first line**. Code before `interrupt()` runs twice, so keep side effects after it.

```python
from langgraph.types import interrupt, Command

class Order(TypedDict):
    item: str
    status: str

def ask_human(s: Order):
    answer = interrupt({"question": f"Buy {s['item']}? (yes/no)"})   # pauses here
    return {"status": "approved" if answer == "yes" else "rejected"}

g = StateGraph(Order)
g.add_node("ask_human", ask_human)
g.add_edge(START, "ask_human"); g.add_edge("ask_human", END)
app = g.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "order-1"}}
result = app.invoke({"item": "iPhone", "status": ""}, config)
print(result["__interrupt__"][0].value)          # {'question': 'Buy iPhone? (yes/no)'}
app.invoke(Command(resume="yes"), config)        # {'item': 'iPhone', 'status': 'approved'}
```

**HITL inside a tool** (chatbot use case): the interrupt pauses the run inside `ToolNode`.
```python
@tool
def buy_stock(symbol: str, qty: int) -> str:
    """Buy shares of a stock."""
    ok = interrupt(f"Approve buying {qty} {symbol}? (yes/no)")
    return f"Bought {qty} {symbol}" if ok == "yes" else "User cancelled the purchase"
```

**Static breakpoints** (good for debugging): `g.compile(checkpointer=..., interrupt_before=["tools"])` stops before every tool run. Resume with `app.invoke(None, config)`.

## 3. RAG inside LangGraph (agentic RAG)
- A RAG **chain** retrieves on every question, even "hi". In LangGraph, the retriever becomes a **tool**, and the LLM decides **whether** and **what** to search. It can also search again with a better query.
- It's the same chat + `ToolNode` graph from the tools section, with a retriever tool added.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(
    PyPDFLoader("islr.pdf").load())
retriever = FAISS.from_documents(chunks, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 4})

@tool
def rag_tool(query: str) -> dict:
    """Search the uploaded PDF. Use this for any question about the document."""
    docs = retriever.invoke(query)
    return {"context": [d.page_content for d in docs],
            "pages": [d.metadata.get("page") for d in docs]}

tools = [rag_tool]
llm_tools = llm.bind_tools(tools)

def chat(s: MessagesState):
    system = SystemMessage("Answer from the rag_tool results. If they don't contain the answer, say you don't know.")
    return {"messages": [llm_tools.invoke([system] + s["messages"])]}

g = StateGraph(MessagesState)
g.add_node("chat", chat)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "chat")
g.add_conditional_edges("chat", tools_condition)
g.add_edge("tools", "chat")
rag_bot = g.compile(checkpointer=InMemorySaver())
```

**One PDF per chat thread** (the Streamlit "upload PDF" feature): keep one retriever per `thread_id`. The tool reads the thread id from the run config. LangChain fills in a `config: RunnableConfig` parameter for you, and the LLM never sees it.
```python
from langchain_core.runnables import RunnableConfig
RETRIEVERS: dict[str, object] = {}        # thread_id -> retriever, filled when a PDF is uploaded

@tool
def pdf_search(query: str, config: RunnableConfig) -> str:
    """Search the PDF uploaded in this chat."""
    retriever = RETRIEVERS.get(config["configurable"]["thread_id"])
    if retriever is None:
        return "No PDF uploaded in this chat."
    return "\n\n".join(d.page_content for d in retriever.invoke(query))
```

## 4. Memory
**Short-term memory** is the messages of one thread, kept by the checkpointer. The problem: the history keeps growing until it costs too much or exceeds the context window. Three fixes:

**a) Trim**: send only the last N tokens to the LLM. The state still keeps everything.
```python
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

def chat(s: MessagesState):
    recent = trim_messages(s["messages"], strategy="last", max_tokens=500,
                           token_counter=count_tokens_approximately,
                           start_on="human", include_system=True)
    return {"messages": [llm.invoke(recent)]}
```

**b) Delete** old messages from the state with `RemoveMessage`. The `add_messages` reducer removes them by id.
```python
from langchain_core.messages import RemoveMessage

def delete_old(s: MessagesState):
    return {"messages": [RemoveMessage(id=m.id) for m in s["messages"][:-6]]}   # keep the last 6
```

**c) Summarize**: replace old messages with a running summary.
```python
class ChatState(MessagesState):
    summary: str

def chat(s: ChatState):
    msgs = s["messages"]
    if s.get("summary"):
        msgs = [SystemMessage(f"Conversation so far: {s['summary']}")] + msgs
    return {"messages": [llm.invoke(msgs)]}

def summarize(s: ChatState):
    prompt = f"Extend this summary: {s.get('summary', '')}\nwith these new messages." if s.get("summary") \
             else "Summarize the conversation above."
    new_summary = llm.invoke(s["messages"] + [HumanMessage(prompt)]).content
    return {"summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in s["messages"][:-2]]}

def should_summarize(s: ChatState) -> Literal["summarize", "__end__"]:
    return "summarize" if len(s["messages"]) > 6 else END

g = StateGraph(ChatState)
g.add_node("chat", chat); g.add_node("summarize", summarize)
g.add_edge(START, "chat")
g.add_conditional_edges("chat", should_summarize)
g.add_edge("summarize", END)
```

**Long-term memory** is facts that last **across threads** (the user's name, preferences, past decisions).
- It's kept in a **Store**, separate from the checkpointer: `InMemoryStore` for learning, `PostgresStore` in production.
- Items are grouped by a **namespace** tuple, for example `("user_1", "memories")`.
- Methods: `put(namespace, key, value)`, `get(namespace, key)`, `search(namespace, query=..., limit=...)`.
- With `index={"embed": ..., "dims": ...}`, `search` finds items by meaning.
- Pass it in with `compile(store=store)`. Nodes receive it through a `store` parameter.

```python
import uuid
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

store = InMemoryStore(index={"embed": OpenAIEmbeddings(model="text-embedding-3-small"), "dims": 1536})

def chat(s: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = (user_id, "memories")
    last = s["messages"][-1].content

    memories = store.search(ns, query=last, limit=3)            # recall
    facts = "\n".join(m.value["fact"] for m in memories)

    if "remember" in last.lower():                              # save (in practice an LLM decides)
        store.put(ns, str(uuid.uuid4()), {"fact": last})

    system = SystemMessage(f"Facts about the user:\n{facts}")
    return {"messages": [llm.invoke([system] + s["messages"])]}

g = StateGraph(MessagesState)
g.add_node("chat", chat)
g.add_edge(START, "chat"); g.add_edge("chat", END)
app = g.compile(checkpointer=InMemorySaver(), store=store)

app.invoke({"messages": [HumanMessage("Remember: I love cricket")]},
           {"configurable": {"thread_id": "t1", "user_id": "u1"}})
app.invoke({"messages": [HumanMessage("What sport do I like?")]},
           {"configurable": {"thread_id": "t2", "user_id": "u1"}})       # new thread, still knows
```

## 5. Subgraphs and multi-agent
**Subgraph**: a compiled graph used inside another graph. Use it to reuse a workflow, keep big graphs readable, or give each agent its own graph.
- **Shared state keys**: add the compiled subgraph directly as a node.
- **Different state**: call the subgraph inside a node function and map the inputs and outputs yourself.

```python
# child graph: translates text
class TState(TypedDict):
    text: str
    translated: str

def translate(s): return {"translated": llm.invoke(f"Translate to Hindi: {s['text']}").content}
child = StateGraph(TState); child.add_node("translate", translate)
child.add_edge(START, "translate"); child.add_edge("translate", END)
translator = child.compile()

# parent with a different state: call the subgraph inside a node
class PState(TypedDict):
    topic: str
    answer: str
    answer_hi: str

def answer(s): return {"answer": llm.invoke(f"One line about {s['topic']}").content}
def to_hindi(s):
    out = translator.invoke({"text": s["answer"]})     # map parent -> child
    return {"answer_hi": out["translated"]}            # map child -> parent

parent = StateGraph(PState)
parent.add_node("answer", answer); parent.add_node("to_hindi", to_hindi)
parent.add_edge(START, "answer"); parent.add_edge("answer", "to_hindi"); parent.add_edge("to_hindi", END)
parent.compile().invoke({"topic": "LangGraph"})
# if PState shared the keys 'text' and 'translated', you could just do parent.add_node("translate", translator)
```

**Multi-agent, supervisor pattern**: one supervisor agent decides which specialist agent handles each part of the task. The simplest version wraps each specialist as a tool of the supervisor.
```python
from langchain.agents import create_agent

@tool
def web_search(q: str) -> str:
    """Search the web."""
    return "Some search results"          # stand-in for a real search tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

research_agent = create_agent(model=llm, tools=[web_search], system_prompt="You research facts.")
math_agent = create_agent(model=llm, tools=[multiply], system_prompt="You do maths.")

@tool
def ask_researcher(task: str) -> str:
    """Delegate research or fact-finding tasks."""
    return research_agent.invoke({"messages": [HumanMessage(task)]})["messages"][-1].content

@tool
def ask_mathematician(task: str) -> str:
    """Delegate any calculation."""
    return math_agent.invoke({"messages": [HumanMessage(task)]})["messages"][-1].content

supervisor = create_agent(model=llm, tools=[ask_researcher, ask_mathematician],
                          system_prompt="Split the task and delegate to your helpers.")
supervisor.invoke({"messages": [HumanMessage("Population of India times 2?")]})
```
- Other patterns:
  - **Network**: any agent can hand off to any other with `Command(goto="agent_b")`.
  - **Hierarchical**: supervisors of supervisors, built from subgraphs.
- There's also the `langgraph-supervisor` package with a ready-made `create_supervisor`.

## 6. Fault tolerance (demo)
- With a checkpointer, every finished super-step is saved. If a node crashes (API timeout, rate limit, bug), `invoke(None, same_config)` **resumes from the failed node**, and nodes that already finished are **not run again**. This saves time and LLM cost.
- `RetryPolicy` retries a flaky node automatically before giving up.

```python
import random
from langgraph.types import RetryPolicy

class S(TypedDict):
    a: str
    b: str

def step_1(s):
    print("step_1 ran")
    return {"a": "done"}

def step_2(s):
    if random.random() < 0.7:
        raise TimeoutError("API timed out")       # fake flaky API
    return {"b": "done"}

g = StateGraph(S)
g.add_node("step_1", step_1)
g.add_node("step_2", step_2, retry_policy=RetryPolicy(max_attempts=3))
g.add_edge(START, "step_1"); g.add_edge("step_1", "step_2"); g.add_edge("step_2", END)
app = g.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "job-1"}}
try:
    app.invoke({"a": "", "b": ""}, config)
except TimeoutError:
    print("crashed, next node:", app.get_state(config).next)    # ('step_2',)

app.invoke(None, config)    # resumes at step_2; "step_1 ran" is not printed again
```

## 7. LangSmith evaluation
- Tracing shows **what happened** in one run. Evaluation measures **how good** the app is across many examples, so you can compare versions (prompt v1 vs v2, k=4 vs k=8).
- The pieces:
  1. A **dataset**: inputs with reference outputs.
  2. A **target**: your app as a function.
  3. **Evaluators**: functions that score each output (exact match, LLM-as-judge).
  4. An **experiment**: one run of the target over the dataset. Compare experiments side by side in the UI.
- Online evaluation and monitoring: score production traces automatically, and add user feedback (thumbs up or down) with `client.create_feedback`.

```python
from langsmith import Client
client = Client()

ds = client.create_dataset("rag-qa")
client.create_examples(
    dataset_id=ds.id,
    inputs=[{"question": "What is overfitting?"}, {"question": "What is a p-value?"}],
    outputs=[{"answer": "Fitting noise in the training data so it doesn't generalise"},
             {"answer": "Probability of data this extreme if the null hypothesis is true"}],
)

def target(inputs: dict) -> dict:
    return {"answer": rag_chain.invoke(inputs["question"])}

class Grade(BaseModel):
    correct: bool

judge = llm.with_structured_output(Grade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return judge.invoke(
        f"Question: {inputs['question']}\nReference: {reference_outputs['answer']}\n"
        f"Answer: {outputs['answer']}\nIs the answer correct?").correct

def concise(outputs: dict) -> bool:
    return len(outputs["answer"].split()) < 100

client.evaluate(target, data="rag-qa", evaluators=[correctness, concise], experiment_prefix="rag-k4")
```
- The `openevals` package has ready-made LLM-as-judge evaluators (correctness, conciseness, hallucination). RAGAS has RAG-specific metrics (faithfulness, context precision).

---

## Self-test
1. Why does an update from 2 parallel nodes to one key fail without a reducer? When is the state merged?
2. When do you use `Send`? When do you use `Command`?
3. What does `interrupt()` return after `Command(resume="yes")`? Why should side effects come after it?
4. Agentic RAG vs a RAG chain: what does the agent decide?
5. Trim vs delete (`RemoveMessage`) vs summarize: what does each one do to the state?
6. Checkpointer vs Store: which one is per thread, and which one works across threads?
7. Subgraph with shared keys vs different keys: how do you add each one?
8. After a crash, how do you resume? Do finished nodes run again?
9. What 4 pieces make up a LangSmith evaluation?
