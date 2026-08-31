from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    openrouter_api_key: str

    vector_store_path: str = "./data/chroma"
    document_store_path: str = "./data/documents.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()