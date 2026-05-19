from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class KnowledgeTopic(StrEnum):
    AGENT_ARCHITECTURE = "agent_architecture"
    AGENT_HARDENING = "agent_hardening"
    CODING_AGENTS = "coding_agents"
    COMPUTER_USE = "computer_use"
    CONTEXT_ENGINEERING = "context_engineering"
    COST_LATENCY = "cost_latency"
    DATA_CONNECTORS = "data_connectors"
    DEPLOYMENT = "deployment"
    GUARDRAILS = "guardrails"
    HUMAN_IN_LOOP = "human_in_loop"
    IDENTITY_ACCESS = "identity_access"
    KNOWLEDGE_GRAPHS = "knowledge_graphs"
    MCP = "mcp"
    MODEL_ROUTING = "model_routing"
    MULTIMODAL_AGENTS = "multimodal_agents"
    MULTI_AGENT = "multi_agent"
    OBSERVABILITY = "observability"
    PLANNING = "planning"
    PROMPT_ENGINEERING = "prompt_engineering"
    PROTOCOLS = "protocols"
    REASONING = "reasoning"
    STATE_RUNTIME = "state_runtime"
    TOOL_USE = "tool_use"
    TOOL_ROUTING = "tool_routing"
    MEMORY = "memory"
    RAG = "rag"
    RETRIEVAL = "retrieval"
    SAFETY = "safety"
    STRUCTURED_OUTPUTS = "structured_outputs"
    WORKFLOW = "workflow"
    EVALUATION = "evaluation"
    FRONTIER_SIGNAL = "frontier_signal"


class KnowledgeCard(BaseModel):
    source_url: HttpUrl
    title: str
    one_sentence: str
    why_it_matters: str
    agent_builder_takeaway: str
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    relevance_score: float = Field(ge=0.0, le=1.0)
    frontier_score: float = Field(ge=0.0, le=1.0)


class KnowledgeIndexEntry(BaseModel):
    source_url: HttpUrl
    source_title: str | None = None
    card_title: str
    one_sentence: str
    why_it_matters: str = ""
    agent_builder_takeaway: str
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    relevance_score: float = Field(ge=0.0, le=1.0)
    frontier_score: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class KnowledgeIndex(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_documents: int = 0
    total_cards: int = 0
    topic_counts: dict[KnowledgeTopic, int] = Field(default_factory=dict)
    entries: list[KnowledgeIndexEntry] = Field(default_factory=list)


class FrontierBrief(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    title: str = "Frontier Agent Brief"
    summary: str
    top_signals: list[str] = Field(default_factory=list)
    agent_builder_moves: list[str] = Field(default_factory=list)
    next_experiments: list[str] = Field(default_factory=list)
    watch_topics: list[KnowledgeTopic] = Field(default_factory=list)
    source_urls: list[HttpUrl] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    source_url: HttpUrl
    title: str | None = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cards: list[KnowledgeCard] = Field(default_factory=list)
    skipped_reason: str | None = None


class AnalysisRunStats(BaseModel):
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    total_documents: int = 0
    analyzed_documents: int = 0
    skipped_documents: int = 0
    total_cards: int = 0

    def finish(self, results: list[AnalysisResult]) -> "AnalysisRunStats":
        self.finished_at = datetime.now(UTC)
        self.total_documents = len(results)
        self.analyzed_documents = sum(1 for item in results if item.cards)
        self.skipped_documents = sum(1 for item in results if not item.cards)
        self.total_cards = sum(len(item.cards) for item in results)
        return self
