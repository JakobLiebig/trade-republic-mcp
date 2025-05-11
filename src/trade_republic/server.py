from fastmcp import FastMCP

from config import config

# Create an MCP server
mcp = FastMCP(
    name="Demo",
    port=config.mcp_port,
)