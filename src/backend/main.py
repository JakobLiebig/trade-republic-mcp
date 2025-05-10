from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.backend.ai import stream_response

app = FastAPI(
    title="Trade Republic Backend",
    description="Backend for the Trade Republic project",
)

@app.get("/chat")
async def chat(message: str):
    return StreamingResponse(
        stream_response(message),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )