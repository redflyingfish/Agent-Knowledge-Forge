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
    recency_policy: str = "ordinary sources: 2026+; official/spec/authority sources: 2025-06-01+"
    topic_expansions: list[TopicExpansion] = Field(default_factory=list)
    frontier_scout_queries: list[str] = Field(default_factory=list)
    source_hub_queries: list[str] = Field(default_factory=list)
    stop_signal_queries: list[str] = Field(default_factory=list)
