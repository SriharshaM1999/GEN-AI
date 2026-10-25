# CampusX playlists: checklist and gaps

Each video's topics are compared with your notes and code.
- `[x]` covered in your notes or code
- `[~]` partly covered
- `[ ]` missing, so revise from the video

The video list is from memory of the playlists. Check the exact titles and order on YouTube.

---

## Playlist 1: Generative AI using LangChain

1. **Intro: what and why LangChain**
   - `[~]` The components overview (models, prompts, chains, memory, indexes, agents) is only in [01_langchain.md](01_langchain.md).
   - `[ ]` Why frameworks are needed: orchestrating many components, model-agnostic code, the "PDF chat app" system design example from the video.
2. **Models**
   - `[x]` LLM vs chat model, temperature, embeddings, cosine similarity.
   - `[ ]` Running an open-source model **locally** with `HuggingFacePipeline` (you only used the Endpoint).
   - `[ ]` Anthropic and Google chat models (same interface).
3. **Prompts**
   - `[x]` PromptTemplate, save/load, messages, ChatPromptTemplate, MessagesPlaceholder, Streamlit UI.
   - `[ ]` The simple **console chatbot loop** that keeps the history list (while True → append Human/AI messages).
   - `[ ]` Loading chat history from a file into MessagesPlaceholder.
4. **Structured output**
   - `[x]` TypedDict, Annotated, Pydantic, JSON schema.
   - `[ ]` `method="json_mode"` vs `"function_calling"`, and which models support which.
5. **Output parsers**
   - `[x]` Str, Json, Structured, Pydantic.
6. **Chains**
   - `[x]` Simple, sequential, parallel, conditional, `get_graph().print_ascii()`.
7. **Runnables: the why**
   - `[ ]` The video builds a toy "nakli LangChain" with its own Runnable base class, to show why a common `invoke` interface lets anything connect to anything. Rewatch it for the idea.
8. **Runnable primitives / LCEL**
   - `[x]` Sequence, Parallel, Passthrough, Lambda, Branch.
   - `[ ]` The `|` operator as LCEL syntax for RunnableSequence (the only one with special syntax). `.batch()` and `.stream()` on any chain.
9. **Document loaders**
   - `[x]` Text, PyPDF, Directory, Web, CSV, lazy_load.
   - `[ ]` Which PDF loader for which PDF (scanned → Unstructured or OCR, tables → PDFPlumber, layout → PyMuPDF).
   - `[ ]` Custom loader via `BaseLoader`.
10. **Text splitters**
    - `[x]` All 4 kinds plus token-based. Your notebook goes further than the video.
11. **Vector stores**
    - `[x]` Chroma CRUD, search, filters, MMR, persistence. Well covered.
    - `[~]` The Chroma *collection* concept, and viewing data with `get(include=["embeddings", ...])`.
12. **Retrievers**
    - `[x]` Vector store retriever, MMR, multi-query, contextual compression with a re-ranker.
    - `[ ]` `WikipediaRetriever`, which shows that a retriever doesn't have to be a vector store.
    - `[ ]` Contextual compression with `LLMChainExtractor`, which trims each chunk down to its relevant sentences.
    - Your [langchain/retrievers/file.py](../langchain/retrievers/file.py) is empty.
13. **What is RAG**
    - `[~]` Only in the LangSmith code.
    - `[ ]` Revise the theory: why not fine-tuning, in-context learning, the 4 steps (indexing, retrieval, augmentation, generation).
14. **RAG project: YouTube chatbot**
    - `[ ]` Not in your notes. It uses `youtube_transcript_api` (`YouTubeTranscriptApi().fetch(video_id)`) → splitter → FAISS → retriever → LCEL chain.
    - `[ ]` The "how to improve" list at the end: UI, evaluation (RAGAS / LangSmith), query rewriting, multimodal, agentic RAG.
15. **Tools**
    - `[x]` Built-in tools, `@tool`, StructuredTool, BaseTool, toolkit.
16. **Tool calling**
    - `[x]` bind_tools → tool_calls → ToolMessage → final answer.
    - `[ ]` **Currency conversion project with `InjectedToolArg`**. Your [2.CurrentConversion.ipynb](<../langchain/Tool Calling/2.CurrentConversion.ipynb>) is empty, so do this one.
17. **AI agents**
    - `[x]` ReAct with hub prompt + AgentExecutor (trip planner), plus the newer `create_agent`.
    - `[ ]` The video's explanation of the internals: the agent picks the next step (Thought, Action), and the AgentExecutor loops, calls the tool and feeds back the Observation, until Final Answer.

## Playlist 2: Agentic AI using LangGraph

1. **Course intro / curriculum**: not needed.
2. **Generative AI vs Agentic AI**
   - `[ ]` Theory only in [02_langgraph.md](02_langgraph.md). Rewatch the "HR recruiter" example.
3. **What is Agentic AI**
   - `[ ]` The 6 traits and the components (brain, orchestrator, tools, memory, supervisor).
4. **LangChain vs LangGraph**
   - `[ ]` The automated hiring workflow example, and the list of problems LangGraph solves (control flow, state, events, fault tolerance, human in the loop, nested workflows, observability).
5. **LangGraph core concepts**
   - `[~]` State and reducers are used in your code.
   - `[ ]` The 5 workflow patterns, and the **Pregel / super-step** execution model.
6. **Sequential workflows**: `[x]` BMI, LLM QnA, prompt chaining.
7. **Parallel workflows**: `[x]` cricket stats, UPSC essay.
   - `[~]` Make sure you remember the partial-update rule and `operator.add`.
8. **Conditional workflows**: `[x]` quadratic, review reply.
9. **Iterative workflows**: `[x]` tweet generator, evaluator and optimizer.
10. **Chatbot**: `[x]` add_messages, InMemorySaver, thread_id.
11. **Persistence**
    - `[x]` get_state, get_state_history, time travel, update_state.
    - `[ ]` The **fault tolerance demo**: a node that crashes or hangs, then you resume with `invoke(None, config)`.
12. **Chatbot UI with Streamlit**: `[x]` basic UI, streaming, resume threads, SQLite persistence.
    - `[ ]` Naming threads (title from the first message) instead of showing raw uuids.
13. **LangSmith**: `[x]` tracing, `@traceable`, project, tags and metadata, RAG v1-v4, agent, LangGraph.
    - `[ ]` Integrating LangSmith into the chatbot, with traces grouped per thread (`metadata={"thread_id": ...}`).
    - `[ ]` Evaluation and datasets.
14. **Tools in LangGraph**: `[x]` ToolNode, tools_condition.
15. **MCP client in LangGraph**: `[x]` MultiServerMCPClient, stdio, async.
    - `[ ]` A **remote** server over `streamable_http`.
    - `[ ]` Async chatbot in Streamlit (an event loop in a background thread).
16. **Videos you still have left** (check the playlist; likely topics):
    - `[ ]` RAG inside LangGraph: the retriever as a tool in the chatbot, uploading a PDF per thread.
    - `[ ]` Human in the loop: `interrupt()` inside a node, resumed with `Command(resume=...)`. Needs a checkpointer.
    - `[ ]` Subgraphs and multi-agent: a compiled graph used as a node.
    - `[ ]` Memory: short-term (trimming or summarizing messages) vs long-term (`Store`, across threads).

## Separate: MCP basics ([mcp/expense-tracker/](../mcp/expense-tracker/))
- `[x]` FastMCP server, `@mcp.tool`, `uv run fastmcp dev inspector`.
- `[ ]` Resources (`@mcp.resource`) and prompts (`@mcp.prompt`). Notes only mention tools.
- `[ ]` Finishing the actual expense tracker (SQLite add/list/summarize tools) and connecting it to Claude Desktop.

---

## Small fixes in your code (worth knowing)
- [13. Tools Integration/file.ipynb](<../langgraph/13. Tools Integration/file.ipynb>) and [14/main.ipynb](<../langgraph/14. Langgraph with MCP/main.ipynb>) use `Annotated[BaseMessage, add_messages]`. It should be `Annotated[list[BaseMessage], add_messages]`.
- In [14/main.ipynb](<../langgraph/14. Langgraph with MCP/main.ipynb>), `chatNode` calls `_initiateChatModel()`, which runs `get_tools()` on **every** call and starts a new server process each time. The practice notebook does it correctly, once at startup.
- [langgraph/practice/file.ipynb](../langgraph/practice/file.ipynb) adds both `add_conditional_edges('get_review', ...)` and `add_edge('get_review', END)`. The plain edge makes END run alongside the router's choice, so drop it.
- [langsmith/langsmith/3_rag_v2.py](../langsmith/langsmith/3_rag_v2.py) has a typo, `LANCHAIN_PROJECT`, so the project name isn't applied.
- [langsmith/langsmith/2_sequential_chain.py](../langsmith/langsmith/2_sequential_chain.py) is missing a comma after the `run_name` value, so the file fails with a SyntaxError.
- [langsmith/langsmith/4_agent.py](../langsmith/langsmith/4_agent.py) has a hardcoded weatherstack API key. Move it to `.env`.
