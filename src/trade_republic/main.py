from src.trade_republic.core.server import mcp
from src.trade_republic.tools.add import add
from src.trade_republic.resources.greeting import get_greeting

if __name__ == "__main__":
    mcp.run(transport="sse")