from pydantic_settings import BaseSettings

class Config(BaseSettings):
    mcp_host: str = "localhost"
    mcp_port: int = 8080

    backend_host: str = "localhost"
    backend_port: int = 8000

    class Config:
        case_sensitive = False

config = Config()
