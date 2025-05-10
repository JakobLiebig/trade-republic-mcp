import asyncio
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai import Agent

from src.config import secrets

# Setup MCP Connection
mcp = MCPServerHTTP(
    url=f"http://localhost:8080/sse",
    sse_read_timeout=30,
)
# Setup model
provider = OpenAIProvider(
    api_key=secrets.openai_api_key
)
model = OpenAIModel(
    model_name="gpt-4.1-mini",
    provider=provider,
)
agent = Agent(
    model=model,
    mcp_servers=[mcp],
    system_prompt="""
You are a highly sofisticated financial advisor.
You have a variaty of tools to fetch data from the user's financial situation and current market prices.
Always keep your awnsers short and concise.

You support markdown formatting.
You support mermaid diagrams.
    """
)
# Setup message history
message_history=[]

async def stream_response(user_message):
    async with agent.run_stream(
        user_message,
        message_history=message_history
    ) as response:
        async for chunk in response.stream_text(delta=True):
            print(chunk, end="", flush=True)
        message_history.extend(response.new_messages())
    print("\n")

async def main():
    async with agent.run_mcp_servers():
        while(True):
            message = input(">")
            await stream_response(message)

if __name__ == '__main__':
    print("To exit press ctrl + c then enter.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bye bye!")
        exit()
