import asyncio
from fastmcp import Client
import json

from src.trade_republic.core.server import mcp
import src.trade_republic.features.query_data

client = Client(mcp)

async def test():
    async with client:
        print(await client.call_tool("query_banking_data", arguments={"start_date": "2024-01-01"}))


if __name__ == "__main__":
    asyncio.run(test())