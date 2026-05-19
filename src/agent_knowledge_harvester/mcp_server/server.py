from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex, KnowledgeIndexEntry

ReportLanguage = Literal["en", "zh", "default"]
MemoryPackKind = Literal["working", "compact", "ultra_compact", "llm_compact", "uncompressed"]


class KnowledgeMCPRepository:
    """Read-only access layer for one Agent Knowledge Forge run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.index_path = run_dir / "03_knowledge_base" / "knowledge_index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"knowledge_index.json not found: {self.index_path}")
        self.index = KnowledgeIndex.model_validate_json(self.index_path.read_text(encoding="utf-8"))

    def summary(self) -> dict[str, object]:
        """Return a compact corpus summary for MCP clients."""
        topics = {
            topic.value if hasattr(topic, "value") else str(topic): count
            for topic, count in self.index.topic_counts.items()
        }
        top_sources = Counter(str(entry.source_url) for entry in self.index.entries).most_common(10)
        return {
            "run_dir": str(self.run_dir),
            "index_path": str(self.index_path),
            "generated_at": self.index.generated_at.isoformat(),
            "total_documents": self.index.total_documents,
            "total_cards": self.index.total_cards,
            "topics": topics,
            "top_sources": [
                {"source_url": source, "cards": count} for source, count in top_sources
            ],
            "recommended_tools": [
                "list_topics",
                "search_agent_knowledge",
                "get_knowledge_card",
                "read_memory_pack",
                "read_human_report",
            ],
        }

    def list_topics(self) -> list[dict[str, object]]:
        """List available topics sorted by card count."""
        rows = []
        for topic, count in self.index.topic_counts.items():
            topic_name = topic.value if hasattr(topic, "value") else str(topic)
            rows.append({"topic": topic_name, "cards": count})
        return sorted(rows, key=lambda item: (-int(item["cards"]), str(item["topic"])))

    def search(
        self,
        query: str = "",
        topic: str | None = None,
        limit: int = 10,
        min_priority: float = 0.0,
    ) -> list[dict[str, object]]:
        """Search cards with lightweight lexical scoring and optional topic filtering."""
        query_terms = tokenize(query)
        normalized_topic = topic.strip().lower() if topic else None
        matches: list[tuple[float, int, KnowledgeIndexEntry]] = []
        for position, entry in enumerate(self.index.entries, start=1):
            topic_values = [item.value for item in entry.topics]
            if normalized_topic and normalized_topic not in topic_values:
                continue
            if entry.priority_score < min_priority:
                continue
            lexical_score = score_entry(entry, query_terms)
            if query_terms and lexical_score <= 0:
                continue
            score = lexical_score + entry.priority_score
            matches.append((score, position, entry))

        matches.sort(key=lambda item: (-item[0], -item[2].priority_score, item[1]))
        return [
            render_card_summary(entry, position, score=score)
            for score, position, entry in matches[: max(1, min(limit, 50))]
        ]

    def get_card(self, card_id: str | int) -> dict[str, object]:
        """Return one full knowledge card by numeric position or card_NNNN id."""
        position = parse_card_position(card_id)
        if position < 1 or position > len(self.index.entries):
            raise ValueError(f"card_id out of range: {card_id}")
        entry = self.index.entries[position - 1]
        return render_full_card(entry, position)

    def read_memory_pack(self, kind: MemoryPackKind = "compact", max_chars: int = 20000) -> str:
        """Read a generated memory pack Markdown file."""
        file_map = {
            "working": "agent_memory_pack.md",
            "compact": "agent_memory_pack.compact.md",
            "ultra_compact": "agent_memory_pack.ultra_compact.md",
            "llm_compact": "agent_memory_pack.llm_compact.md",
            "uncompressed": "agent_memory_pack.uncompressed.md",
        }
        path = self.run_dir / "04_memory_packs" / file_map[kind]
        return read_markdown_artifact(path, max_chars=max_chars)

    def read_human_report(
        self,
        language: ReportLanguage = "default",
        max_chars: int = 30000,
    ) -> str:
        """Read the generated human learning report."""
        file_map = {
            "default": "frontier_learning_report.md",
            "en": "frontier_learning_report.en.md",
            "zh": "frontier_learning_report.zh.md",
        }
        path = self.run_dir / "05_human_report" / file_map[language]
        return read_markdown_artifact(path, max_chars=max_chars)


def create_knowledge_mcp_server(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    """Create an MCP server exposing one run directory as a read-only knowledge base."""
    repository = KnowledgeMCPRepository(run_dir)
    server = FastMCP(
        "agent-knowledge-forge",
        host=host,
        port=port,
        instructions=(
            "Read-only MCP server for Agent Knowledge Forge outputs. Use it to retrieve "
            "source-grounded agent-development knowledge cards, compact memory packs, and "
            "human learning reports from a completed local run."
        ),
    )

    @server.tool()
    def get_corpus_summary() -> dict[str, object]:
        """Summarize the loaded Agent Knowledge Forge corpus."""
        return repository.summary()

    @server.tool()
    def list_topics() -> list[dict[str, object]]:
        """List available knowledge topics and card counts."""
        return repository.list_topics()

    @server.tool()
    def search_agent_knowledge(
        query: str = "",
        topic: str | None = None,
        limit: int = 10,
        min_priority: float = 0.0,
    ) -> list[dict[str, object]]:
        """Search source-grounded agent-development knowledge cards."""
        return repository.search(
            query=query,
            topic=topic,
            limit=limit,
            min_priority=min_priority,
        )

    @server.tool()
    def get_knowledge_card(card_id: str) -> dict[str, object]:
        """Read one full card. Use IDs returned by search_agent_knowledge."""
        return repository.get_card(card_id)

    @server.tool()
    def read_memory_pack(kind: MemoryPackKind = "compact", max_chars: int = 20000) -> str:
        """Read an agent memory pack by kind."""
        return repository.read_memory_pack(kind=kind, max_chars=max_chars)

    @server.tool()
    def read_human_report(
        language: ReportLanguage = "default",
        max_chars: int = 30000,
    ) -> str:
        """Read a human report in default, en, or zh form."""
        return repository.read_human_report(language=language, max_chars=max_chars)

    return server


def tokenize(text: str) -> list[str]:
    """Tokenize mixed English/Chinese text for simple local retrieval."""
    normalized = text.lower()
    ascii_terms = re.findall(r"[a-z0-9_+#.-]{2,}", normalized)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    return ascii_terms + chinese_terms


def score_entry(entry: KnowledgeIndexEntry, query_terms: list[str]) -> float:
    """Score one card with weighted lexical matches."""
    if not query_terms:
        return 1.0
    fields = {
        "title": entry.card_title,
        "claim": entry.one_sentence,
        "takeaway": entry.agent_builder_takeaway,
        "why": entry.why_it_matters,
        "implementation": " ".join(entry.implementation_notes),
        "topics": " ".join(topic.value for topic in entry.topics),
        "evidence": " ".join(entry.evidence[:3]),
    }
    weights = {
        "title": 3.0,
        "topics": 2.5,
        "claim": 2.0,
        "takeaway": 2.0,
        "implementation": 1.5,
        "why": 1.0,
        "evidence": 0.8,
    }
    score = 0.0
    for field, value in fields.items():
        haystack = value.lower()
        for term in query_terms:
            if term in haystack:
                score += weights[field]
    return score


def parse_card_position(card_id: str | int) -> int:
    """Parse card ids like 1, '1', 'card_0001', or 'A1'."""
    if isinstance(card_id, int):
        return card_id
    normalized = card_id.strip().lower()
    match = re.search(r"(\d+)", normalized)
    if not match:
        raise ValueError(f"invalid card_id: {card_id}")
    return int(match.group(1))


def render_card_summary(
    entry: KnowledgeIndexEntry,
    position: int,
    score: float,
) -> dict[str, object]:
    """Render a compact search result."""
    return {
        "card_id": f"card_{position:04d}",
        "rank_score": round(score, 3),
        "title": entry.card_title,
        "claim": entry.one_sentence,
        "agent_builder_takeaway": entry.agent_builder_takeaway,
        "topics": [topic.value for topic in entry.topics],
        "priority_score": entry.priority_score,
        "source_url": str(entry.source_url),
    }


def render_full_card(entry: KnowledgeIndexEntry, position: int) -> dict[str, object]:
    """Render one full card for MCP clients."""
    return {
        "card_id": f"card_{position:04d}",
        "title": entry.card_title,
        "source_url": str(entry.source_url),
        "source_title": entry.source_title,
        "topics": [topic.value for topic in entry.topics],
        "priority_score": entry.priority_score,
        "relevance_score": entry.relevance_score,
        "frontier_score": entry.frontier_score,
        "claim": entry.one_sentence,
        "why_it_matters": entry.why_it_matters,
        "agent_builder_takeaway": entry.agent_builder_takeaway,
        "implementation_notes": entry.implementation_notes,
        "evidence": entry.evidence,
    }


def read_markdown_artifact(path: Path, max_chars: int) -> str:
    """Read a Markdown artifact with a clear truncation marker for MCP clients."""
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    text = path.read_text(encoding="utf-8")
    budget = max(1000, min(max_chars, 200000))
    if len(text) <= budget:
        return text
    return (
        text[:budget].rstrip()
        + f"\n\n[Truncated at {budget} characters. Increase max_chars for more content.]\n"
    )
