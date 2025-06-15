from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    openai_api_key: str
    discourse_url: str
    discourse_search_filters: str
    discourse_cookie: str
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings()
