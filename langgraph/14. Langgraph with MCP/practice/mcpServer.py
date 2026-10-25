from fastmcp import FastMCP

mcp = FastMCP(name = 'calculator server')

@mcp.tool
def calculator(a: int, b: int, c: str) -> int: 
    """performs necessary operation"""
    return a + b + 10


if __name__ == '__main__':
    mcp.run(transport='stdio')