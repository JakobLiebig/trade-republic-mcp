# Trade Republic MCP

A Model Context Protocol Server for Trade Republic.

## Prerequisites

- Python 3.13 or higher
- Git
- UV package manager

## Installation

1. Clone the repository:
```bash
git clone ... trade-republic-mcp
cd trade-republic-mcp
```

2. Download missing files, create simplified dbs:
```bash
curl -L "https://github.com/Wenzhi-Ding/Std_Security_Code/raw/refs/heads/main/isin/company_name.pq?download=" > ./data/company_name.pq
```


3. Install dependencies:
```bash
uv sync
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the MCP Server

Start the Model Context Protocol server:
```bash
uv run -m src.trade_republic.main
```

### Running the Backend Server

Start the backend server:
```bash
uv run -m src.backend.main
```

### Running the Test Client

Run the test client:
```bash
uv run -m src.client
```

## Project Structure

- `src/trade_republic/` - MCP server implementation
- `src/backend/` - Backend server implementation
- `src/client.py` - Test client implementation
