from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # Qdrant settings
    qdrant_url: str = "http://qdrant-4shqdxl2cwx2ppiygxnzbbcl.57.131.144.251.sslip.io"
    qdrant_api_key: str = "Il14wLQHJJM5SncYCJzReR26VtxfyvPn"
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
