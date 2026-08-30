from fastmcp import FastMCP
import sys

mcp = FastMCP(name = 'SriharshaMCPServer')


@mcp.tool
async def add(a: int, b: int) -> int: 
    """ Use this tool if you want to add 2 numbers """
    print("HEY.......! Add tool got called", file=sys.stderr)
    return a + b + 5


@mcp.tool
async def minus(a: int, b: int) -> int: 
    """ Use this tool if you want to subtract 2 numbers """
    print("HEY.......! Minus tool got called", file=sys.stderr)
    return a - b


@mcp.tool
async def multiply(a: int, b: int) -> int: 
    """ Use this tool if you want to multiply 2 numbers """
    print("HEY.......! Multiply tool got called", file=sys.stderr)
    return a * b


@mcp.tool
async def divide(a: int, b: int) -> float: 
    """ Use this tool if you want to divide 2 numbers """
    print("HEY.......! Divide tool got called", file=sys.stderr)
    return a / b


if __name__ == '__main__':
    mcp.run(transport="stdio")
