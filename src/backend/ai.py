import asyncio
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent

from src.config import secrets

# Setup MCP Connection
mcp = MCPServerHTTP(
    url="http://localhost:8080/sse",
)
# Setup model
provider = OpenAIProvider(
    api_key=secrets.openai_api_key
)
model = OpenAIModel(
    model_name="gpt-4o-mini",
    provider=provider,
)
agent = Agent(
    model=model,
    mcp_servers=[mcp],
)
# Setup message history
message_history=[]

async def stream_response(user_message):
    async with agent.run_mcp_servers():
        async with agent.run_stream(
            user_message,
            message_history=message_history
        ) as response:
            async for chunk in response.stream_text(delta=True):
                yield chunk
            message_history.extend(response.new_messages())
        yield "\n"
