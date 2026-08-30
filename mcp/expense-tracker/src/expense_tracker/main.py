import random
from fastmcp import FastMCP

mcp = FastMCP(name="Sriharha Server")

@mcp.tool # -> makes a python function a mcp tool.
def roll_dice(n_dice: int = 1)->list[int]:
    """Roll n_dice 6-sided dice and return the results"""
    return [random.randint(1,6) for _ in range(n_dice)]

@mcp.tool
def add_numbers(a:float, b:float)-> float:
    """Add 2 numbers"""
    return a + b

if __name__ == '__main__':
    mcp.run()

# cmd to run -> uv run fastmcp dev inspector src/expense_tracker/main.py
