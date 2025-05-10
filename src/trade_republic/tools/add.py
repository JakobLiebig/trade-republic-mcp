from src.trade_republic.core.server import mcp

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b + 1