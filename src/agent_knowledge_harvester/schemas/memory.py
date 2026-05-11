from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl

from agent_knowledge_harvester.schemas.analysis import KnowledgeTopic


class AgentMemoryEntry(BaseModel):
    source_url: HttpUrl
    source_title: str | None = None
    card_title: str
    claim: str
    agent_move: str
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    priority_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    generated_at: datetime


class AgentMemoryPack(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_indexes: list[str] = Field(default_factory=list)
    after: datetime | None = None
    total_input_entries: int = 0
    retained_entries: int = 0
    dropped_duplicates: int = 0
    dropped_by_time: int = 0
    dropped_by_priority: int = 0
    entries: list[AgentMemoryEntry] = Field(default_factory=list)


class CompactMemoryPack(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_pack: str | None = None
    budget: str = "compact"
    scope: str = "frontier agent development"
    core_rules: list[str] = Field(default_factory=list)
    current_patterns: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    retrieval_pointers: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)
    memory_operations: list[str] = Field(default_factory=list)


class RetrievalManifestEntry(BaseModel):
    memory_id: str
    source_url: HttpUrl
    source_title: str | None = None
    card_title: str
    claim: str
    agent_move: str
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    priority_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    frontier_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class RetrievalManifest(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_index: str
    total_entries: int = 0
    entries: list[RetrievalManifestEntry] = Field(default_factory=list)
    topic_index: dict[str, list[str]] = Field(default_factory=dict)
    source_index_map: dict[str, list[str]] = Field(default_factory=dict)
