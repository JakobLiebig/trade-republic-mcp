from pydantic_settings import BaseSettings

class Config(BaseSettings):
    backend_host: str = "localhost"
    backend_port: int = 8000

    mcp_host: str = "localhost"
    mcp_port: int = 8080

    class Config:
        case_sensitive = False


class Secrets(BaseSettings):
    mistral_api_key: str

    class Config:
        case_sensitive = False
        env_file = ".env"

config = Config()
secrets = Secrets()