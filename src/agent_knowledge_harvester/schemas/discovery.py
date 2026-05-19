from pydantic import BaseModel, Field


class TopicExpansion(BaseModel):
    topic: str
    seed_terms: list[str] = Field(default_factory=list)
    adjacent_terms: list[str] = Field(default_factory=list)
    authority_queries: list[str] = Field(default_factory=list)
    implementation_queries: list[str] = Field(default_factory=list)
    risk_queries: list[str] = Field(default_factory=list)


class SearchQueryPlan(BaseModel):
    name: str = "frontier-agent-development-query-plan"
    recency_policy: str = "ordinary sources: 2025+; unknown-date non-authority sources need review"
    topic_expansions: list[TopicExpansion] = Field(default_factory=list)
    frontier_scout_queries: list[str] = Field(default_factory=list)
    source_hub_queries: list[str] = Field(default_factory=list)
    stop_signal_queries: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Normalized result returned by a search API provider."""

    url: str
    title: str = ""
    snippet: str = ""
    source: str
    query: str
    rank: int
    published_at: str | None = None
    language: str | None = None
    raw: dict[str, object] = Field(default_factory=dict)


class DiscoveryStats(BaseModel):
    """Operational counters for a discovery run."""

    provider: str
    planned_queries: int
    executed_queries: int
    requested_results_per_query: int
    raw_results: int
    unique_urls: int
    duplicate_results: int
    failed_queries: int
    year: int


class SearchDiscoveryReport(BaseModel):
    """Durable artifact for automatic URL discovery."""

    query_plan: SearchQueryPlan
    stats: DiscoveryStats
    results: list[SearchResult] = Field(default_factory=list)
    candidate_urls: list[str] = Field(default_factory=list)
    failed_queries: dict[str, str] = Field(default_factory=dict)
