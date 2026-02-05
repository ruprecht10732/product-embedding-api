from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # Qdrant settings
    qdrant_url: str = "https://qdrant-m4804ssokwsggcgkgws0wcoc.salestainable.nl:443"
    qdrant_api_key: str = ""
    collection_name: str = "houthandel_products"
    
    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    
    # Output
    output_dir: str = "output"
    
    class Config:
        env_prefix = "EMBED_"


@lru_cache
def get_settings() -> Settings:
    return Settings()
