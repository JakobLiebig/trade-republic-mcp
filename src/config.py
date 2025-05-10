from pydantic_settings import BaseSettings

class Secrets(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"

class Config(BaseSettings):
    mcp_port: int = 8080
    backend_port: int = 8000

    class Config:
        env_file = ".env"


secrets = Secrets()
config = Config()