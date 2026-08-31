from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str

    database_url: str = (
        "postgresql+psycopg://postgres:postgres@db:5432/knowledge"
    )

    vector_store_path: str = "./data/chroma"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()