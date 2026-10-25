# LangChain: quick revision

Source folders: [langchain/](../langchain/), [langsmith/](../langsmith/) (RAG lives there)

Every snippet assumes `from dotenv import load_dotenv; load_dotenv()` has run.

---

## 1. Why LangChain
- A framework for building LLM apps. Same interface for every provider, so you can swap OpenAI for Anthropic or HuggingFace in one line.
- Main pieces: **Models, Prompts, Chains, Memory, Indexes (loaders, splitters, vector stores, retrievers), Agents**.

## 2. Models ([langchain/models/](../langchain/models/))
- **LLM** (`OpenAI`): string in, string out. Old API, mostly unused now.
- **Chat model** (`ChatOpenAI`): list of messages in, `AIMessage` out. Use `.content` for the text. The response also carries `response_metadata` (token usage) and `tool_calls`.
- Params: `temperature` (0 means deterministic, above 1 is creative), `max_completion_tokens`, `top_p`.
- Open source: `HuggingFaceEndpoint` (hosted API, needs `HUGGINGFACEHUB_API_TOKEN`) wrapped in `ChatHuggingFace`. `HuggingFacePipeline` runs the model locally.
- **Embeddings**: `embed_query(text)` returns one vector. `embed_documents([texts])` returns a list of vectors. `OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)` lets you shrink the vector size.
- Similarity: `cosine_similarity([query_vec], doc_vecs)` then `argmax`. This was the mini "semantic search" project.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_completion_tokens=300)
print(llm.invoke("Capital of India?").content)

# open source, hosted
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
hf = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation"))

# embeddings + similarity search
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
emb = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
texts = ["Kohli is a batsman", "Dhoni is a captain"]
scores = cosine_similarity([emb.embed_query("Who is Kohli?")], emb.embed_documents(texts))
print(texts[np.argmax(scores)])
```

## 3. Prompts ([langchain/prompts/](../langchain/prompts/))
- **Static prompt**: the user types the whole thing. **Dynamic prompt**: a template plus values the user picks (the research paper summarizer Streamlit UI).
- `PromptTemplate(template=..., input_variables=[...], validate_template=True)`. Why not an f-string? It validates the variables, can be saved and loaded (`template.save('template.json')`, `load_prompt(...)`), and it's a Runnable, so `template | llm` works.
- Messages: `SystemMessage` (role/rules), `HumanMessage` (user), `AIMessage` (model). For a chatbot with history, keep appending them to a list.
- `ChatPromptTemplate([("system", "You are a {domain} expert"), MessagesPlaceholder("messages"), ("human", "...")])`.
  - Use **tuples** `("system", "...{x}")`. If you build `SystemMessage("...{x}")` objects directly, the `{x}` is not filled in.
  - `MessagesPlaceholder` inserts a whole list of past messages (chat history).

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

template = PromptTemplate(template="Summarize {paper} in {style} style",
                          input_variables=["paper", "style"], validate_template=True)
template.save("template.json")
chain = load_prompt("template.json") | llm

# console chatbot with history
history = [SystemMessage("You are a helpful assistant")]
while (q := input("You: ")) != "exit":
    history.append(HumanMessage(q))
    reply = llm.invoke(history)
    history.append(AIMessage(reply.content))
    print("AI:", reply.content)

chat_prompt = ChatPromptTemplate([
    ("system", "You are a {domain} expert"),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}"),
])
chat_prompt.invoke({"domain": "ML", "chat_history": history, "query": "Explain attention"})
```

## 4. Structured output ([langchain/output/structuredoutput/](../langchain/output/structuredoutput/))
- `model.with_structured_output(Schema)` returns data, not text. Works only with models that support it (OpenAI and most big providers).
- Three ways to write the schema:
  - **TypedDict**: type hints only, no validation. `Annotated[str, "description"]` guides the LLM. `Optional[...]` marks optional fields. Result is a `dict`.
  - **Pydantic** `BaseModel` + `Field(description=..., ge=, le=)`: validates the data. Result is an object (`.model_dump()`, `.model_dump_json()`).
  - **JSON schema** dict: when you share schemas with other languages. Result is a `dict`.
- `Literal["pos", "neg"]` restricts a field to fixed values.
- `method="function_calling"` (default for OpenAI) or `"json_mode"`.

```python
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

class ReviewTD(TypedDict):
    summary: Annotated[str, "A brief summary"]
    sentiment: Annotated[Literal["pos", "neg"], "Sentiment"]
    pros: Annotated[Optional[list[str]], "List of pros"]

class Review(BaseModel):
    summary: str = Field(description="A brief summary")
    sentiment: Literal["pos", "neg"]
    rating: int = Field(ge=1, le=5)

llm.with_structured_output(ReviewTD).invoke(review_text)["summary"]   # dict
r = llm.with_structured_output(Review).invoke(review_text)            # Review object
r.rating, r.model_dump()
```

## 5. Output parsers ([langchain/output/outputParsers/](../langchain/output/outputParsers/))
Use these when the model can't do structured output (for example open-source models), or to turn output into a string.
- `StrOutputParser`: `AIMessage` to `str`. Needed to chain model → next prompt.
- `JsonOutputParser`: any JSON, but **you can't force the keys**.
- `StructuredOutputParser` + `ResponseSchema`: forces the keys, **no type validation**.
- `PydanticOutputParser(pydantic_object=Person)`: forces the keys **and** validates types. The best of the parsers.
- Pattern: `partial_variables={"format_instruction": parser.get_format_instructions()}` puts the format rules into the prompt. Chain: `template | model | parser`.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

class Person(BaseModel):
    name: str
    age: int = Field(gt=18)

parser = PydanticOutputParser(pydantic_object=Person)
# JsonOutputParser() / StructuredOutputParser.from_response_schemas([ResponseSchema(name="fact_1", description="...")])
template = PromptTemplate(
    template="Make up a {place} person\n{format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)
person = (template | hf | parser).invoke({"place": "Sri Lankan"})   # Person object
```

## 6. Chains ([langchain/chains/](../langchain/chains/))
- **Simple**: `prompt | model | parser`.
- **Sequential**: `t1 | model | parser | t2 | model | parser` (detailed report, then a summary).
- **Parallel**: `RunnableParallel({"notes": c1, "quiz": c2}) | merge_prompt | model | parser`.
- **Conditional**: classify first (Pydantic parser), then `RunnableBranch((cond, chain), ..., default)`.
- `chain.get_graph().print_ascii()` draws the chain.

```python
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
parser = StrOutputParser()

# sequential
report = PromptTemplate.from_template("Detailed report on {topic}")
summary = PromptTemplate.from_template("5 line summary of:\n{text}")
chain = report | llm | parser | summary | llm | parser
chain.invoke({"topic": "black holes"})
chain.get_graph().print_ascii()

# parallel
notes = PromptTemplate.from_template("5 notes on {topic}") | llm | parser
quiz = PromptTemplate.from_template("5 questions on {topic}") | llm | parser
merge = PromptTemplate.from_template("Merge {notes} and {quiz}")
(RunnableParallel({"notes": notes, "quiz": quiz}) | merge | llm | parser).invoke({"topic": "cricket"})

# conditional: classify_chain returns a Review object with .sentiment
branch = RunnableBranch(
    (lambda r: r.sentiment == "positive", positive_reply_chain),
    (lambda r: r.sentiment == "negative", negative_reply_chain),
    RunnableLambda(lambda r: "could not find sentiment"),
)
(classify_chain | branch).invoke({"input": "This phone is great"})
```

## 7. Runnables / LCEL ([langchain/runnables/](../langchain/runnables/))
- **Why**: early LangChain had many components with different method names. A **Runnable** is a common interface with `invoke`, `batch`, `stream` (plus async versions), and any Runnable can be connected to another one.
- Two kinds: **task-specific** runnables (prompt, model, parser, retriever) and **primitives** that wire them together:
  - `RunnableSequence(a, b, c)` is the same as `a | b | c` (that `|` syntax is **LCEL**).
  - `RunnableParallel({...})`: same input to every branch, returns a dict.
  - `RunnablePassthrough()`: returns its input unchanged. Keeps the original joke next to its explanation.
  - `RunnableLambda(fn)`: turns any Python function into a Runnable (word count, `format_docs`).
  - `RunnableBranch((cond, runnable), default)`: if/else inside a chain.

```python
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

joke_chain = RunnableSequence(PromptTemplate.from_template("Joke about {topic}"), llm, parser)
final = joke_chain | RunnableParallel({
    "joke": RunnablePassthrough(),                                   # keep the joke as is
    "words": RunnableLambda(lambda text: len(text.split())),         # plain function
    "explanation": PromptTemplate.from_template("Explain: {text}") | llm | parser,
})
final.invoke({"topic": "AI"})      # {'joke': ..., 'words': ..., 'explanation': ...}

# summarize only if the report is long
RunnableBranch((lambda x: len(x.split()) > 300, summary | llm | parser), RunnablePassthrough())

# every runnable has the same methods
joke_chain.batch([{"topic": "cats"}, {"topic": "dogs"}])
for chunk in joke_chain.stream({"topic": "cats"}): print(chunk, end="")
```

## 8. Document loaders ([langchain/docloaders/](../langchain/docloaders/))
- Every loader returns `list[Document]`, and each `Document` has `page_content` and `metadata`.
- `TextLoader`, `PyPDFLoader` (one Document per page, needs `pypdf`), `CSVLoader` (one per row), `WebBaseLoader` (one URL or a list; uses BeautifulSoup), `DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)`.
- `load()` reads everything into memory at once. `lazy_load()` is a generator, one Document at a time. Use it for many or large files.
- Other PDF loaders for harder files: `PyMuPDFLoader`, `UnstructuredPDFLoader`, `PDFPlumberLoader`. You can write your own by subclassing `BaseLoader`.

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, WebBaseLoader, DirectoryLoader)

docs = TextLoader("cricket.txt").load()
pages = PyPDFLoader("dl-curriculum.pdf").load()            # one Document per page
rows = CSVLoader(file_path="Social_Network_Ads.csv").load() # one per row
web = WebBaseLoader("https://example.com").load()
print(pages[0].page_content[:100], pages[0].metadata)

loader = DirectoryLoader("resources", glob="**/*.pdf", loader_cls=PyPDFLoader)
for doc in loader.lazy_load():                              # one at a time, low memory
    print(doc.metadata["source"])
```

## 9. Text splitters ([langchain/Splitters/](../langchain/Splitters/))
- Why split: model context limits, better embeddings, better retrieval.
- `chunk_size`, `chunk_overlap` (about 10-20%). `split_text(str)` returns strings. `split_documents(docs)` keeps the metadata.
- Kinds:
  - **Length based**: `CharacterTextSplitter`. One separator, so chunks can come out too big.
  - **Text structure based**: `RecursiveCharacterTextSplitter`. Tries `\n\n` → `\n` → space → characters. **The default choice.**
  - **Document structure based**: `.from_language(Language.PYTHON)`, `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`, `RecursiveJsonSplitter`.
  - **Token based**: `.from_tiktoken_encoder(...)`, `TokenTextSplitter`.
  - **Semantic**: `SemanticChunker(embeddings, breakpoint_threshold_type="percentile" | "standard_deviation")`. Starts a new chunk where the meaning shifts.
- Production recipe: split by structure first, then run the recursive splitter to cap the size.

```python
from langchain_text_splitters import (CharacterTextSplitter, RecursiveCharacterTextSplitter,
                                      MarkdownHeaderTextSplitter, Language)
from langchain_experimental.text_splitter import SemanticChunker

CharacterTextSplitter(separator="", chunk_size=200, chunk_overlap=0).split_documents(pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)       # Documents in, Documents out (metadata kept)

RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=300).split_text(code)
RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=100)

md_docs = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2")]).split_text(markdown_text)
final = splitter.split_documents(md_docs)      # structure first, then cap the size

SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
                breakpoint_threshold_amount=1).create_documents([text])
```

## 10. Vector stores ([langchain/VectorStore/VectorStore.ipynb](../langchain/VectorStore/VectorStore.ipynb))
- Store embeddings plus text plus metadata, and search by meaning.
- CRUD: `add_documents(docs, ids=...)` (custom ids mean re-runs upsert instead of duplicating), `get`, `get_by_ids`, `update_documents`, `delete(ids=...)`.
- Search: `similarity_search(q, k, filter={...})`, `similarity_search_with_score` (distance, **lower is better**), `similarity_search_with_relevance_scores` (0 to 1, **higher is better**), `max_marginal_relevance_search` (relevant **and** diverse), `similarity_search_by_vector`.
- Chroma filters: `$eq $ne $gt $gte $lt $lte $in $nin $and $or`, and `where_document={"$contains": "word"}` to filter on the text.
- Persist: `Chroma(collection_name, embedding_function, persist_directory)`. Reopen it with the **same embedding model**.
- Stores: InMemory (tests), Chroma (local), FAISS (`FAISS.from_documents`, fast library), pgvector, Pinecone, Qdrant, MongoDB Atlas.

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document

store = Chroma(collection_name="demo", embedding_function=OpenAIEmbeddings(),
               persist_directory="chroma_db")
docs = [Document(page_content="Mangoes are sweet", metadata={"category": "fruit", "year": 2024})]
store.add_documents(docs, ids=["doc_0"])                     # same id again = upsert
store.update_document("doc_0", Document(page_content="Mangoes are tropical", metadata={"category": "fruit"}))
store.get(include=["documents", "metadatas"])
store.delete(ids=["doc_0"])

store.similarity_search("tropical fruit", k=2,
                        filter={"$and": [{"category": "fruit"}, {"year": {"$gte": 2023}}]})
store.similarity_search_with_relevance_scores("fruit", k=3)  # (doc, 0-1 score)
store.max_marginal_relevance_search("mango", k=3, fetch_k=8, lambda_mult=0.4)
store.similarity_search("apple", k=3, where_document={"$contains": "iPhone"})

from langchain_community.vectorstores import FAISS
faiss_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
```

## 11. Retrievers
- A retriever does one thing: query in, `list[Document]` out (`.invoke`). It's a Runnable, so it goes inside chains.
- `store.as_retriever(search_type="similarity" | "mmr" | "similarity_score_threshold", search_kwargs={"k": 4, ...})`.
- **Hybrid**: `EnsembleRetriever([BM25Retriever, vector_retriever], weights=[0.5, 0.5])`. Keywords plus meaning.
- **Multi-query**: `MultiQueryRetriever.from_llm(retriever, llm)`. The LLM rewrites the question several ways and the results are merged.
- **Contextual compression**: `ContextualCompressionRetriever(base_compressor=..., base_retriever=...)`. The compressor can be a re-ranker (`CrossEncoderReranker`) or `LLMChainExtractor`, which keeps only the relevant sentences.
- **Parent document**: search small child chunks, return the big parent.
- **Indexing API**: `index(docs, SQLRecordManager, store, cleanup="incremental" | "full", source_id_key="source")`. Skips unchanged chunks and deletes stale ones.
- Not vector based: `WikipediaRetriever(top_k_results=2, lang="en")`. It searches Wikipedia directly.

```python
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8})
retriever.invoke("vegetables for eyesight")

from langchain_community.retrievers import WikipediaRetriever, BM25Retriever
WikipediaRetriever(top_k_results=2, lang="en").invoke("Virat Kohli")

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

hybrid = EnsembleRetriever(retrievers=[BM25Retriever.from_documents(docs), retriever], weights=[0.5, 0.5])
multi = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
compressed = ContextualCompressionRetriever(base_compressor=LLMChainExtractor.from_llm(llm),
                                            base_retriever=retriever)
compressed.invoke("What is photosynthesis?")
```

## 12. RAG ([langsmith/langsmith/3_rag_v1.py](../langsmith/langsmith/3_rag_v1.py))
- Why RAG: LLMs don't know private or recent data. Fine-tuning is costly and slow. RAG = **in-context learning with retrieved context**.
- 4 steps:
  1. **Indexing**: load → split → embed → store.
  2. **Retrieval**: the retriever fetches the top-k chunks.
  3. **Augmentation**: put the context and the question into the prompt ("answer ONLY from context, else say you don't know").
  4. **Generation**: the LLM answers.
- CampusX project: **YouTube chatbot**. `youtube_transcript_api` gets the transcript → splitter → FAISS → the chain below.
- Ways to improve it: better chunking, MMR or multi-query retrieval, re-ranking, evaluation (RAGAS / LangSmith), and agentic RAG (the retriever as a tool).

```python
from youtube_transcript_api import YouTubeTranscriptApi

# 1. indexing
transcript = " ".join(s.text for s in YouTubeTranscriptApi().fetch("Gfr50f6ZBvo", languages=["en"]))
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([transcript])
retriever = FAISS.from_documents(chunks, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 4})

# 2-4. retrieval, augmentation, generation
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context. If it's not there, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])
format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
rag_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
}) | prompt | llm | StrOutputParser()
rag_chain.invoke("What is discussed about aliens?")
```

## 13. Tools ([langchain/Tools/](../langchain/Tools/))
- A tool is a function the LLM can ask to run. It has a **name, a description (the docstring), and an args schema (the type hints)**.
- Built-in: `DuckDuckGoSearchRun`, `DuckDuckGoSearchResults`, `ShellTool`, Wikipedia, Python REPL...
- Three ways to write your own:
  - `@tool`: needs a docstring and type hints. Inspect it with `.name`, `.description`, `.args`.
  - `StructuredTool.from_function(func, name, description, args_schema=PydanticModel)`.
  - `BaseTool` subclass with `name`, `description`, `args_schema`, and `_run`.
- **Toolkit**: a class whose `get_tools()` returns related tools.

```python
from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Type

DuckDuckGoSearchRun().invoke("latest IPL news")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
multiply.invoke({"a": 3, "b": 5}); multiply.name, multiply.description, multiply.args

class MulInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

mul2 = StructuredTool.from_function(func=lambda a, b: a * b, name="multiply",
                                    description="Multiply two numbers", args_schema=MulInput)

class MulTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MulInput
    def _run(self, a: int, b: int) -> int:
        return a * b

class MathToolkit:
    def get_tools(self): return [multiply, mul2]
```

## 14. Tool calling ([langchain/Tool Calling/1.file.ipynb](<../langchain/Tool Calling/1.file.ipynb>))
- `llm.bind_tools([multiply])`: the LLM learns the tool schemas.
- Flow: `HumanMessage` → `AIMessage.tool_calls` (name, args, id) → `tool.invoke(tool_call)` returns a `ToolMessage` → send all the messages back → final answer.
- The **LLM never runs the tool**. It only says which tool to call and with what arguments. Your code runs it.
- **`InjectedToolArg`** (CampusX currency-conversion project): `rate: Annotated[float, InjectedToolArg]` hides an argument from the LLM, and your code fills it in from the earlier tool's result before calling the tool. Your [2.CurrentConversion.ipynb](<../langchain/Tool Calling/2.CurrentConversion.ipynb>) is still empty.

```python
llm_with_tools = llm.bind_tools([multiply])

messages = [HumanMessage("multiply 2 with 6")]
ai_msg = llm_with_tools.invoke(messages)        # ai_msg.tool_calls = [{'name': 'multiply', 'args': {...}, 'id': ...}]
messages.append(ai_msg)
for call in ai_msg.tool_calls:
    messages.append(multiply.invoke(call))      # passing a ToolCall returns a ToolMessage
print(llm_with_tools.invoke(messages).content)

# InjectedToolArg: the LLM doesn't fill conversion_rate, your code does
import json
from langchain_core.tools import InjectedToolArg

@tool
def get_conversion_factor(base: str, target: str) -> float:
    """Get the currency conversion rate from base to target"""
    return 85.3          # in the video this calls a currency API

@tool
def convert(amount: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert amount using the conversion rate"""
    return amount * conversion_rate

llm_ct = llm.bind_tools([get_conversion_factor, convert])
messages = [HumanMessage("Convert 10 USD to INR")]
ai_msg = llm_ct.invoke(messages); messages.append(ai_msg)
for call in ai_msg.tool_calls:
    if call["name"] == "get_conversion_factor":
        result = get_conversion_factor.invoke(call)
        rate = json.loads(result.content)
        messages.append(result)
    elif call["name"] == "convert":
        call["args"]["conversion_rate"] = rate              # inject it yourself
        messages.append(convert.invoke(call))
llm_ct.invoke(messages).content
```

## 15. Agents ([langchain/agents/agents.ipynb](../langchain/agents/agents.ipynb), [tripPlanner.ipynb](../langchain/endToendProject/tripPlanner.ipynb))
- Chain: the steps are fixed. Agent: **the LLM decides the steps** in a loop (reason → act → observe).
- **ReAct** (the CampusX version): `create_react_agent(llm, tools, prompt=hub "hwchase17/react")` + `AgentExecutor(agent, tools, verbose=True, max_iterations=5)`. The text format is Thought / Action / Action Input / Observation, then Final Answer. The **agent** decides the next step. The **AgentExecutor** runs the loop and calls the tools. Weak spot: multi-argument tools break, because Action Input is a single string.
- `create_tool_calling_agent` + `MessagesPlaceholder("agent_scratchpad")`: native tool calls, so multi-argument tools work.
- LangChain v1 `create_agent(model, tools, system_prompt, checkpointer, response_format, middleware, context_schema)`. Under the hood it's a LangGraph graph.
  - Memory: `checkpointer=InMemorySaver()` + `{"configurable": {"thread_id": ...}}`.
  - `response_format=Pydantic` gives you `result["structured_response"]`.
  - `ToolRuntime` + `context=` passes hidden values to tools (user id, secrets).
  - Streaming `stream_mode`: `"updates"`, `"values"`, `"messages"` (one token at a time).
  - Middleware: call limits, summarization, retry, fallback, PII, `HumanInTheLoopMiddleware` (resume with `Command(resume=...)`), custom `@before_model`, `@dynamic_prompt`, `@wrap_tool_call`.
  - Tools that **raise** crash the whole run. Return an error string instead, or catch errors with `@wrap_tool_call`.

```python
# ReAct (classic, CampusX)
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
search = DuckDuckGoSearchRun()
agent = create_react_agent(llm=llm, tools=[search], prompt=hub.pull("hwchase17/react"))
executor = AgentExecutor(agent=agent, tools=[search], verbose=True, max_iterations=5)
executor.invoke({"input": "Plan a 3-day trip to Goa"})["output"]

# LangChain v1
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Ctx:
    user_id: str

@tool
def get_my_orders(runtime: ToolRuntime[Ctx]) -> str:
    """Get the current user's orders."""
    return f"orders of {runtime.context.user_id}"      # the LLM never sees user_id

agent = create_agent(
    model=llm, tools=[multiply, get_my_orders],
    system_prompt="Use tools instead of guessing.",
    checkpointer=InMemorySaver(),
    context_schema=Ctx,
    middleware=[ModelCallLimitMiddleware(run_limit=5)],
)
config = {"configurable": {"thread_id": "user-1"}}
res = agent.invoke({"messages": [{"role": "user", "content": "What did I order?"}]},
                   config, context=Ctx(user_id="u1"))
res["messages"][-1].content

for step in agent.stream({"messages": [{"role": "user", "content": "3*4?"}]}, config, stream_mode="updates"):
    print(step)
```

---

## Self-test
1. Why does `ChatPromptTemplate([SystemMessage("You are {x}")])` not fill in `{x}`?
2. JsonOutputParser vs StructuredOutputParser vs PydanticOutputParser: which validates types?
3. What does `RunnablePassthrough` do inside `RunnableParallel` in a RAG chain?
4. `load()` vs `lazy_load()`?
5. `similarity_search_with_score` vs `..._with_relevance_scores`: which direction is "better"?
6. MMR: what do `fetch_k` and `lambda_mult` control?
7. Who runs the tool in tool calling, the LLM or your code? What does `InjectedToolArg` solve?
8. Agent vs AgentExecutor? Why does ReAct fail with multi-argument tools?
