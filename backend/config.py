from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "BizIntel RAG API"
    app_env: str = "development"

    # Supabase
    supabase_url: str
    auth_api_key: str
    database_api_key: str
    storage_bucket: str = "documents"


    # Answer generation
    groq_api_key: str
    groq_model_name: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: int = 45
    groq_max_attempts: int = 4
    max_output_tokens: int = 800

    # CORS
    allowed_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )

    # Upload validation
    max_upload_bytes: int = 5 * 1024 * 1024
    max_filename_length: int = 120
    allowed_extensions: str = (
    ".txt,.md,.markdown,.csv,.json,.pdf,.docx,"
    ".xlsx,.xls,.pptx,.ppt,.doc,.html,.htm,.xml,.rtf"
)

    # Chroma
    #chroma_path: str = "./chroma_vector_db"

    # Vector database provider.
    # TODO: Set VECTOR_DB_PROVIDER=qdrant on Render after creating a

    # Qdrant Cloud cluster and adding QDRANT_CLOUD_URL/QDRANT_API_KEY.
    vector_db_provider: str = "qdrant"
    qdrant_cloud_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "enterprise_knowledge"
    qdrant_vector_size: int = 1024

    # Embeddings 
    # TODO: Set EMBEDDING_PROVIDER=jina on Render after adding JINA_API_KEY.
    embedding_provider: str = "jina"
    # Previous local embedding model reference only; active Jina model is jina_embedding_model below.
    # embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    jina_api_key: Optional[str] = None
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_timeout_seconds: int = 60

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    retrieve_n: int = 5
    top_k: int = 3
    reranker_threshold: float = 0.35
    semantic_fallback_threshold: float = 0.25
    keyword_fallback_threshold: float = 0.5
    # Previous local reranker model reference only; active Jina model is jina_reranker_model below.
    # reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # TODO: Set RERANKER_PROVIDER=jina on Render after adding JINA_API_KEY.
    # Use "local" to keep the existing cross-encoder behavior.
    # Use "none" to skip reranking and rely on vector similarity.
    reranker_provider: str = "jina"
    jina_reranker_model: str = "jina-reranker-v2-base-multilingual"

    # Conversations
    conversation_history_turns: int = 3
    max_history_chars: int = 12000

    # Rate limiting
    ask_rate_limit: int = 20
    upload_rate_limit: int = 5
    rate_limit_window_seconds: int = 60

    max_prompt_length: int = 4000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "ALLOWED_ORIGINS cannot be empty"
            )

        return value

    @property
    def allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {
            extension.strip().lower()
            for extension in self.allowed_extensions.split(",")
            if extension.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
settings = get_settings()


