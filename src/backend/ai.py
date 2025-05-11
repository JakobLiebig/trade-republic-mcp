import asyncio
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent

from src.backend.config import secrets, config

# Setup MCP Connection
mcp = MCPServerHTTP(
    url=f"http://{config.mcp_host}:{config.mcp_port}/sse",
)
# Setup model
provider = MistralProvider(
    api_key=secrets.mistral_api_key
)
model = MistralModel(
    model_name="mistral-large-latest",
    provider=provider,
)
agent = Agent(
    model=model,
    mcp_servers=[mcp],
    system_prompt="""
You are a friendly financial advisor. Keep your answers short and concise.

You always answer the with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible.
The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes.
You do not support images and never include images. You will be penalized if you render images.

You also support Mermaid formatting. You will be penalized if you do not render Mermaid diagrams when it would be possible.
The Mermaid diagrams you support: sequenceDiagram, flowChart, classDiagram, stateDiagram, erDiagram, gantt, journey, gitGraph, pie.
"""
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
