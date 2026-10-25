# LangGraph + MCP: revision notes

## 1. What is MCP?
- **Model Context Protocol** is a standard way to expose tools, resources and prompts to LLM apps.
- **Server**: defines the tools ([localMCPServer.py](localMCPServer.py)).
- **Client**: connects to the server, finds its tools, and calls them ([main.ipynb](main.ipynb)).
- Write a tool once and any MCP client (LangGraph, Claude Desktop, Cursor…) can use it.

## 2. MCP server (FastMCP)
```python
from fastmcp import FastMCP
mcp = FastMCP(name='SriharshaMCPServer')

@mcp.tool
async def add(a: int, b: int) -> int: 
   # On the server side, async def in FastMCP is optional. A plain def works too. Async only helps when the tool itself does I/O, like API calls, database queries, or file reads.
    """Use this tool if you want to add 2 numbers"""   # docstring becomes the tool description the LLM sees
    return a + b

if __name__ == '__main__':
    mcp.run(transport="stdio")
```
- `@mcp.tool`: type hints become the input schema, and the docstring becomes the description.
- **Log with `print(..., file=sys.stderr)`**. With stdio, **stdout is the protocol channel**, so a normal `print` corrupts the messages.

## 3. Transports
| Transport | How it works | When to use it |
|---|---|---|
| `stdio` | Client starts the server as a subprocess and talks over stdin/stdout | Local tools |
| `streamable_http` / `sse` | Server runs on its own at a URL | Remote or shared servers |

## 4. MCP client (`langchain_mcp_adapters`)
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_CONFIG = {
    "math": {                          # one entry per server
        "transport": "stdio",
        "command": sys.executable,     # the venv's python
        "args": ["/abs/path/localMCPServer.py"],
    },
    # "remote": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
}
client = MultiServerMCPClient(MCP_CONFIG)
tools = await client.get_tools()       # converts MCP tools into LangChain StructuredTools
```
- `MultiServerMCPClient` connects to **several servers at once** and merges their tools.
- `get_tools()` is **async**, so you have to `await` it.
- The tools it returns are ordinary LangChain tools, so they work with `bind_tools` and `ToolNode`.

## 5. Wiring it into LangGraph
```python
chatModel = ChatOpenAI().bind_tools(tools)   # LLM learns the tool schemas
toolNode  = ToolNode(tools)                  # runs the tools the LLM asks for

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)  # tool_calls? -> 'tools', else END
graph.add_edge('tools', 'chat_node')                       # tool result goes back to the LLM

await workflow.ainvoke({'messages': ["result of add: 1 , 2"]})
```
Flow: user → `chat_node` (AIMessage with `tool_calls`) → `tools` (ToolMessage) → `chat_node` (final answer).

## 6. Gotchas
- **MCP tools are async only**: use `await workflow.ainvoke(...)`. Calling `.invoke()` fails.
- Inside an async node, prefer `await chatModel.ainvoke(...)`.
- **Call `get_tools()` once at startup**, not inside `chatNode`. Each call opens a new stdio session, which starts a new server process.
- State type: `messages: Annotated[list[BaseMessage], add_messages]` (use `list[...]`, not a bare `BaseMessage`).
- Use an **absolute path** for the server script in `args`.
- The `add` tool returns `a + b + 5`. That's a handy check: a "wrong" answer shows the result came from the MCP tool, not the LLM's own math.

## 7. FastMCP CLI commands
Activate the venv first (`source venv/bin/activate`). `list` and `call` start the server with the `fastmcp` command, so it has to be on your PATH.

| Command | What it does |
|---|---|
| `fastmcp run server.py` | Run the server (stdio by default) |
| `fastmcp run server.py -t http --port 8000` | Run it over HTTP at `http://localhost:8000/mcp` |
| `fastmcp run server.py:mcp` | Run a specific server object from the file |
| `fastmcp dev inspector server.py` | Open the **MCP Inspector** web UI to test tools by hand |
| `fastmcp list server.py` | List the server's tools (`--resources`, `--prompts`, `--input-schema`, `--json`) |
| `fastmcp call server.py calculator a=1 b=2 c=add` | Call a tool from the terminal (prints `{"result": 13}`) |
| `fastmcp call server.py tool --input-json '{"a":1}'` | Call it with JSON args |
| `fastmcp inspect server.py` | Summary of tools, resources and prompts (`-f mcp -o out.json` for a report) |
| `fastmcp install claude-code server.py` | Register the server in Claude Code (also `claude-desktop`, `cursor`, `gemini-cli`) |
| `fastmcp install mcp-json server.py` | Print the config JSON to paste into a client |
| `fastmcp version` | Show FastMCP and MCP versions |

- `run` also accepts a URL (proxies a remote server) or an `mcp.json` config.
- Pass arguments to your own script after `--`: `fastmcp run server.py -- --debug`.
