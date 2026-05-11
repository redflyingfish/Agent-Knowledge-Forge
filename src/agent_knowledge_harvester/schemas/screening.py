from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class ScreeningDecision(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


class SourceCandidate(BaseModel):
    url: HttpUrl
    title: str
    summary: str = ""
    source_kind: str = "github_repository"
    discovered_from: str | None = None
    author: str | None = None
    repo_stars: int | None = None
    repo_forks: int | None = None
    owner_followers: int | None = None
    pushed_at: datetime | None = None
    topics: list[str] = Field(default_factory=list)


class LLMScreeningJudgment(BaseModel):
    decision: ScreeningDecision
    agent_relevance: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    classic_value: float = Field(ge=0.0, le=1.0)
    memory_action: str = "skip"
    reasoning: str = ""
    concerns: list[str] = Field(default_factory=list)
    model: str | None = None
    prompt_tokens_estimate: int = 0
    completion_tokens_estimate: int = 0


class ScreenedSource(BaseModel):
    candidate: SourceCandidate
    decision: ScreeningDecision
    overall_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    authority_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    duplicate_of: str | None = None
    reasons: list[str] = Field(default_factory=list)
    pre_llm_decision: ScreeningDecision | None = None
    pre_llm_overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_judgment: LLMScreeningJudgment | None = None


class ScreeningReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_candidates: int = 0
    accepted: int = 0
    review: int = 0
    rejected: int = 0
    llm_enabled: bool = False
    llm_judged: int = 0
    selected_urls: list[HttpUrl] = Field(default_factory=list)
    sources: list[ScreenedSource] = Field(default_factory=list)
