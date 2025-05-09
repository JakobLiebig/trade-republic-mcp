# Trade Republic MCP
A Model Context Protocol Server for Trade Republic.

Setup
```
git clone ... trade-republic-mcp
cd trade-republic-mcp
uv pip install --all
```

Create .env and add these variables:
```
OPENAI_API_KEY
```

Run the server
```
python3 src/trade-republic/server.py
```

Run the client
```
python3 src/client.py
```
