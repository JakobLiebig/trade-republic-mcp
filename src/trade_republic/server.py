from fastmcp import FastMCP

from trade_republic.config import config

# Create an MCP server
mcp = FastMCP(
    name="Demo",
    port=config.mcp_port,
)