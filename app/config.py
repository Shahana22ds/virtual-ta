from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    openai_api_key: str
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings()
