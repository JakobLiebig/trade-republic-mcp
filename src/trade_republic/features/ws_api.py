import websockets
import asyncio

import json

# from src.trade_republic.core.server import mcp

# from src.config import secrets

# ISIN and MCID lookups


class WsApiConnection:
    def __init__(self, verbose: bool = False):
        self.ws = None
        self.sub_id = 0
        self.verbose = verbose

    async def connect(self):
        self.ws = await websockets.connect("wss://api.traderepublic.com")
        await self.ws.send("connect 30")

        if self.verbose:
            print("> connect 30")

        msg = await self.ws.recv()
        if self.verbose:
            print("< " + msg)

        if msg != "connected":
            raise Exception("Failed to connect to the API")

    async def subscribe(self, payload: dict):
        message = "sub " + str(self.sub_id) + " " + json.dumps(payload)
        if self.verbose:
            print("> " + message)

        await self.ws.send(message)
        response = await self.ws.recv()

        if self.verbose:
            print("< " + response)

        code = response.split(" ")[1]
        if code == "E":
            raise ValueError(f"Failed to subscribe to {payload}: {response}")

        response = response.removeprefix(f"{self.sub_id} A ")

        self.sub_id += 1
        return json.loads(response)

    async def fetch(self, instrument_id: str, type: str) -> dict:
        return await self.subscribe({"type": type, "id": instrument_id})

    async def search(
        self, query: str, asset_type: str = "stock", page: int = 1, page_size: int = 10
    ) -> dict:
        search_parameters = {
            "q": query,
            "filter": [{"key": "type", "value": asset_type}],
            "page": page,
            "pageSize": page_size,
        }
        return await self.subscribe(
            {"type": "neonSearchTags", "data": search_parameters}
        )
