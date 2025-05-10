from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.backend.ai import stream_response

app = FastAPI(
    title="Trade Republic Backend",
    description="Backend for the Trade Republic project",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Allow requests from your frontend origin
    allow_origins=["*"],
    # Alternatively, allow all origins with:
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

async def stream_wrapper(message: str):
    async for chunk in stream_response(message):
        yield f"event: message\ndata: {chunk}\n\n"

@app.post("/chat")
async def chat(message: str):
    return StreamingResponse(stream_response(message), media_type="text/event-stream")


@app.get("/health")
@app.options("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
