from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

AnalysisStage = Literal["screening", "extraction", "validation", "linking"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AKH_",
        extra="ignore",
    )

    # Preferred generic OpenAI-compatible LLM settings.
    # Use the fast model for cheap screening and the pro model for detailed analysis.
    llm_provider: str = "deepseek"
    llm_api_key: str | None = None
    llm_base_url: HttpUrl = "https://api.deepseek.com/v1"
    llm_fast_model: str = "deepseek-chat"
    llm_pro_model: str = "deepseek-reasoner"
    llm_fast_api_key: str | None = None
    llm_pro_api_key: str | None = None
    llm_fast_base_url: HttpUrl | None = None
    llm_pro_base_url: HttpUrl | None = None

    # Backward-compatible DeepSeek aliases. Prefer AKH_LLM_* in new deployments.
    deepseek_api_key: str | None = None
    deepseek_base_url: HttpUrl = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    embedding_provider: str = Field(default="local-bge-m3")
    embedding_model: str = Field(default="BAAI/bge-m3")

    # Optional web search provider settings for automatic discovery.
    search_provider: str | None = None
    search_api_key: str | None = None
    search_base_url: HttpUrl | None = None
    search_results_per_query: int = 5
    search_max_queries: int = 80
    search_concurrency: int = 3
    search_timeout_seconds: float = 30.0
    tavily_api_key: str | None = None
    brave_search_api_key: str | None = None
    serpapi_api_key: str | None = None
    exa_api_key: str | None = None

    request_timeout_seconds: float = 120.0
    ingestion_timeout_seconds: float = 45.0
    max_markdown_chars: int = 120_000
    min_markdown_chars: int = 300
    user_agent: str = (
        "Agent-Knowledge-Forge/0.1 "
        "(research crawler; contact: local-dev@example.invalid)"
    )

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    ingested_dir: Path = Path("data/ingested")
    logs_dir: Path = Path("logs")

    @property
    def fast_model(self) -> str:
        return self.llm_fast_model or self.deepseek_model

    @property
    def pro_model(self) -> str:
        return self.llm_pro_model or self.deepseek_model

    @property
    def fast_api_key(self) -> str | None:
        return self.llm_fast_api_key or self.llm_api_key or self.deepseek_api_key

    @property
    def pro_api_key(self) -> str | None:
        return self.llm_pro_api_key or self.llm_api_key or self.deepseek_api_key

    @property
    def fast_base_url(self) -> HttpUrl:
        return self.llm_fast_base_url or self.llm_base_url or self.deepseek_base_url

    @property
    def pro_base_url(self) -> HttpUrl:
        return self.llm_pro_base_url or self.llm_base_url or self.deepseek_base_url

    def model_for_stage(self, stage: AnalysisStage) -> str:
        if stage == "screening":
            return self.fast_model
        return self.pro_model

    def api_key_for_stage(self, stage: AnalysisStage) -> str | None:
        if stage == "screening":
            return self.fast_api_key
        return self.pro_api_key

    def base_url_for_stage(self, stage: AnalysisStage) -> HttpUrl:
        if stage == "screening":
            return self.fast_base_url
        return self.pro_base_url


settings = Settings()
