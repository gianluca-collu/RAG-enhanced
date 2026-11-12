from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for RAGSimple application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Azure OpenAI
    openai_url: str = Field(..., description="Azure OpenAI endpoint URL")
    openai_key: str = Field(..., description="Azure OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o",
        description="LlamaIndex-supported model alias for metadata; Azure routing uses openai_deployment_name",
    )
    openai_deployment_name: str = Field(default="gpt-4o", description="OpenAI deployment name")
    openai_api_version: str = Field(default="2024-10-21", description="Azure OpenAI API version")

    # Azure AI Search
    search_url: str = Field(..., description="Azure AI Search endpoint URL")
    search_key: str = Field(..., description="Azure AI Search API key")
    index_name: str = Field(..., min_length=1, description="AI Search index name")

    # Embeddings
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model name")
    embedding_deployment_name: str = Field(default="text-embedding-ada-002", description="Embedding deployment name")
    embedding_dimensions: int = Field(default=1536, description="Embedding vector dimensions")

    # Document ingestion
    document_intelligence_endpoint: Optional[str] = Field(default=None, description="Azure Document Intelligence endpoint")
    document_intelligence_key: Optional[str] = Field(default=None, description="Azure Document Intelligence key")
    enable_vision: bool = Field(
        default=False,
        description="Extract and describe document figures with Azure OpenAI Vision during ingestion",
    )
    vision_deployment: Optional[str] = Field(default=None, description="Vision-capable Azure OpenAI deployment")
    describe_images: bool = True
    describe_diagrams: bool = True
    vision_max_workers: int = 4
    min_visual_area_ratio: float = 0.01
    enable_semantic_metadata: bool = Field(default=False, description="Generate semantic metadata with Azure OpenAI")
    table_max_tokens: int = 800
    exclude_tables_inside_figures: bool = True
    exclude_layout_grids: bool = True
    exclude_page_boilerplate: bool = True
    boilerplate_action: str = "remove"

    # Application Settings
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    max_context_length: int = Field(default=4000, description="Maximum context length")
    chunk_size: int = Field(default=1000, description="Chunk size for document splitting")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")

    # Chat Settings
    chat_temperature: float = Field(default=0.7, description="LLM temperature")
    chat_max_tokens: int = Field(default=2000, description="Max tokens in response")
    enable_memory: bool = Field(default=True, description="Enable chat memory")
    max_memory_messages: int = Field(default=20, description="Max messages to keep in memory")

    # Search Settings
    search_top_k: int = Field(default=5, description="Number of top results to retrieve")
    min_relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional retrieval score cutoff; calibrate it on an evaluation set",
    )
    hybrid_search: bool = Field(default=True, description="Enable hybrid search")

    @property
    def search_endpoint(self) -> str:
        return self.search_url

    @property
    def search_index_name(self) -> str:
        return self.index_name

    @property
    def embedding_api_version(self) -> str:
        return self.openai_api_version

    @property
    def azure_api_version(self) -> str:
        return self.openai_api_version

    @field_validator("openai_url", "search_url", mode="before")
    @classmethod
    def validate_urls(cls, v):
        """Validate URLs start with https:// and strip trailing slash."""
        if not isinstance(v, str):
            raise ValueError("URL must be a string")
        if not v.startswith("https://"):
            raise ValueError("URLs must start with https://")
        return v.rstrip("/")

    @field_validator("chat_temperature", mode="before")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 1."""
        v = float(v) if isinstance(v, str) else v
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @staticmethod
    def from_env(env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment file."""
        return Config(_env_file=env_file or ".env")


def load_config(env_file: Optional[str] = None) -> Config:
    """Convenience function to load configuration."""
    return Config.from_env(env_file)
