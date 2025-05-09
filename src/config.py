from pydantic_settings import BaseSettings

class Secrets(BaseSettings):
    openai_api_key: str

    class Config:
        case_sensitive = False
        env_file = ".env"

secrets = Secrets()