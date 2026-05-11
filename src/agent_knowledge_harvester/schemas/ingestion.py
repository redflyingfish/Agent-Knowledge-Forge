from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SourceKind(StrEnum):
    URL = "url"
    GITHUB_TRENDING = "github_trending"


class FetchMethod(StrEnum):
    JINA_READER = "jina_reader"
    CRAWL4AI = "crawl4ai"


class UrlTarget(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    url: HttpUrl
    source_kind: SourceKind = SourceKind.URL
    tags: list[str] = Field(default_factory=list)
    discovered_from: str | None = None


class FetchStats(BaseModel):
    method: FetchMethod
    source_url: HttpUrl
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_code: int | None = None
    elapsed_ms: int = 0
    input_chars: int = 0
    output_chars: int = 0
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: str | None = None


class RawDocument(BaseModel):
    source_url: HttpUrl
    final_url: HttpUrl | None = None
    method: FetchMethod
    title: str | None = None
    markdown: str
    stats: FetchStats
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("markdown")
    @classmethod
    def markdown_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("markdown cannot be blank")
        return value


class CleanDocument(BaseModel):
    source_url: HttpUrl
    final_url: HttpUrl | None = None
    title: str | None = None
    markdown: str
    summary_hint: str
    technical_signal_score: float = Field(ge=0.0, le=1.0)
    stats: FetchStats
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    target: UrlTarget
    raw: RawDocument | None = None
    clean: CleanDocument | None = None
    success: bool
    error: str | None = None


class CrawlRunStats(BaseModel):
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    total_targets: int = 0
    succeeded: int = 0
    failed: int = 0
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def finish(self, results: list[IngestionResult]) -> "CrawlRunStats":
        self.finished_at = datetime.now(UTC)
        self.total_targets = len(results)
        self.succeeded = sum(1 for item in results if item.success)
        self.failed = self.total_targets - self.succeeded
        self.estimated_tokens = sum(
            item.clean.stats.estimated_tokens
            for item in results
            if item.clean is not None
        )
        self.estimated_cost_usd = sum(
            item.clean.stats.estimated_cost_usd
            for item in results
            if item.clean is not None
        )
        return self
