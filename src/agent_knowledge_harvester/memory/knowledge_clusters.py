from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)
from agent_knowledge_harvester.schemas.memory import KnowledgeCluster
from agent_knowledge_harvester.utils.files import stable_slug, write_json
from agent_knowledge_harvester.utils.text import clean_display_text

T = TypeVar("T")


def build_knowledge_clusters(
    index: KnowledgeIndex,
    source_index: Path,
    max_items_per_cluster: int = 6,
) -> list[KnowledgeCluster]:
    """Group knowledge cards into topic clusters for survey-style review."""
    grouped: dict[str, list[KnowledgeIndexEntry]] = defaultdict(list)
    for entry in index.entries:
        primary_topic = entry.topics[0].value if entry.topics else "agent_architecture"
        grouped[primary_topic].append(entry)

    clusters = [
        build_cluster(topic, entries, index.generated_at, source_index, max_items_per_cluster)
        for topic, entries in grouped.items()
    ]
    return sorted(clusters, key=lambda item: item.avg_priority_score, reverse=True)


def build_cluster(
    primary_topic: str,
    entries: list[KnowledgeIndexEntry],
    generated_at: datetime,
    source_index: Path,
    max_items: int,
) -> KnowledgeCluster:
    ranked = sorted(entries, key=lambda item: item.priority_score, reverse=True)
    selected = ranked[:max_items]
    topic_values = unique_values(topic for entry in ranked for topic in entry.topics)
    source_urls = unique_values(entry.source_url for entry in selected)
    evidence_count = sum(len(entry.evidence) for entry in ranked)
    avg_priority = round(sum(entry.priority_score for entry in ranked) / len(ranked), 3)
    primary = (
        ranked[0].topics[0]
        if ranked and ranked[0].topics
        else KnowledgeTopic.AGENT_ARCHITECTURE
    )
    title = cluster_title(primary.value)
    return KnowledgeCluster(
        cluster_id=stable_slug(f"{primary_topic} {title}", max_len=40),
        title=title,
        primary_topic=primary,
        topics=topic_values,
        source_urls=source_urls,
        card_titles=[entry.card_title for entry in selected],
        top_claims=[entry.one_sentence for entry in selected],
        agent_moves=[entry.agent_builder_takeaway for entry in selected],
        evidence_count=evidence_count,
        entry_count=len(ranked),
        avg_priority_score=avg_priority,
        generated_at=generated_at,
        source_index=str(source_index),
    )


def write_knowledge_clusters(
    index_path: Path,
    out_dir: Path | None = None,
) -> list[KnowledgeCluster]:
    index = KnowledgeIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    clusters = build_knowledge_clusters(index, source_index=index_path)
    target_dir = out_dir or index_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        target_dir / "knowledge_clusters.json",
        {
            "total_clusters": len(clusters),
            "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
        },
    )
    (target_dir / "knowledge_clusters.md").write_text(
        render_knowledge_clusters(clusters),
        encoding="utf-8",
    )
    return clusters


def render_knowledge_clusters(clusters: list[KnowledgeCluster]) -> str:
    lines = [
        "# Knowledge Clusters",
        "",
        "Survey-style topic clusters for browsing, gap analysis, and next-run planning.",
        "",
        f"Total clusters: {len(clusters)}",
        "",
    ]
    if not clusters:
        lines.append("No clusters available.")
        return "\n".join(lines).strip() + "\n"

    for cluster in clusters:
        topics = ", ".join(topic.value for topic in cluster.topics)
        lines.extend(
            [
                f"## {cluster.title}",
                "",
                f"- Cluster ID: {cluster.cluster_id}",
                f"- Primary topic: {cluster.primary_topic.value}",
                f"- Topics: {topics}",
                f"- Entries: {cluster.entry_count}",
                f"- Avg priority: {cluster.avg_priority_score:.3f}",
                f"- Evidence snippets: {cluster.evidence_count}",
                "",
                "### Top Claims",
                "",
            ]
        )
        lines.extend(f"- {clean_display_text(claim)}" for claim in cluster.top_claims)
        lines.extend(["", "### Agent Moves", ""])
        lines.extend(f"- {clean_display_text(move)}" for move in cluster.agent_moves)
        lines.extend(["", "### Sources", ""])
        lines.extend(f"- {url}" for url in cluster.source_urls)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def cluster_title(primary_topic: str) -> str:
    return {
        "agent_architecture": "Agent Architecture",
        "agent_hardening": "Agent Hardening",
        "claude_code": "Claude Code",
        "coding_agents": "Coding Agents",
        "computer_use": "Computer Use",
        "context_engineering": "Context Engineering",
        "cost_latency": "Cost And Latency",
        "data_connectors": "Data Connectors",
        "deployment": "Deployment",
        "guardrails": "Guardrails",
        "hardness": "Task Hardness",
        "human_in_loop": "Human-In-The-Loop",
        "identity_access": "Identity And Access",
        "knowledge_graphs": "Knowledge Graphs",
        "mcp": "MCP And Tool Protocols",
        "model_routing": "Model Routing",
        "multimodal_agents": "Multimodal Agents",
        "multi_agent": "Multi-Agent Coordination",
        "observability": "Observability And Evaluation",
        "planning": "Planning",
        "prompt_engineering": "Prompt Engineering",
        "protocols": "Protocols",
        "reasoning": "Reasoning",
        "state_runtime": "State And Runtime",
        "tool_use": "Tool Use",
        "tool_routing": "Tool Routing",
        "memory": "Memory",
        "rag": "RAG",
        "retrieval": "Retrieval",
        "safety": "Safety",
        "skills": "Agent Skills",
        "structured_outputs": "Structured Outputs",
        "workflow": "Workflow Design",
        "openclaw": "OpenClaw",
        "evaluation": "Evaluation",
        "frontier_signal": "Frontier Signals",
    }.get(primary_topic, primary_topic.replace("_", " ").title())


def unique_values(items: Iterable[T]) -> list[T]:
    seen: set[str] = set()
    output = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output
