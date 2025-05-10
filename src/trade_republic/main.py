from src.trade_republic.core.server import mcp

import src.trade_republic.features.test

if __name__ == "__main__":
    mcp.run(transport="sse")