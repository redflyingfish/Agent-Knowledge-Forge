import json
from datetime import datetime
from pathlib import Path

from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex, KnowledgeIndexEntry
from agent_knowledge_harvester.schemas.memory import KnowledgeChunk
from agent_knowledge_harvester.utils.files import stable_slug, write_json
from agent_knowledge_harvester.utils.text import clean_display_text


def build_knowledge_chunks(index: KnowledgeIndex, source_index: Path) -> list[KnowledgeChunk]:
    """Build embedding-ready chunks from source-grounded knowledge cards."""
    return [
        build_knowledge_chunk(item, position, index.generated_at, source_index)
        for position, item in enumerate(index.entries, start=1)
    ]


def build_knowledge_chunk(
    item: KnowledgeIndexEntry,
    position: int,
    generated_at: datetime,
    source_index: Path,
) -> KnowledgeChunk:
    """Create one RAG-ready chunk while preserving source evidence."""
    slug = stable_slug(
        " ".join([str(item.source_url), item.card_title, item.one_sentence]),
        max_len=48,
    )
    return KnowledgeChunk(
        chunk_id=f"kc{position:04d}-{slug}",
        source_url=item.source_url,
        source_title=item.source_title,
        card_title=item.card_title,
        text=render_chunk_text(item),
        topics=item.topics,
        primary_topic=item.topics[0] if item.topics else None,
        priority_score=item.priority_score,
        relevance_score=item.relevance_score,
        frontier_score=item.frontier_score,
        evidence=item.evidence,
        retrieval_queries=build_retrieval_queries(item),
        generated_at=generated_at,
        source_index=str(source_index),
    )


def render_chunk_text(item: KnowledgeIndexEntry) -> str:
    """Render a compact chunk that is useful without external context."""
    topics = ", ".join(topic.value for topic in item.topics) or "agent_development"
    lines = [
        f"Title: {item.card_title}",
        f"Source: {item.source_url}",
        f"Topics: {topics}",
        f"Claim: {item.one_sentence}",
        f"Agent builder takeaway: {item.agent_builder_takeaway}",
    ]
    if item.evidence:
        lines.append("Evidence:")
        lines.extend(f"- {evidence}" for evidence in item.evidence[:3])
    return clean_display_text("\n".join(lines))


def build_retrieval_queries(item: KnowledgeIndexEntry) -> list[str]:
    """Generate lightweight query hints for file-search or vector retrieval."""
    topic_terms = [topic.value.replace("_", " ") for topic in item.topics[:4]]
    queries = [
        item.card_title,
        item.one_sentence,
        f"{item.card_title} agent development",
    ]
    queries.extend(f"{term} agent pattern" for term in topic_terms)
    return unique_strings(queries)[:8]


def write_knowledge_chunks(index_path: Path, out_dir: Path | None = None) -> list[KnowledgeChunk]:
    index = KnowledgeIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    chunks = build_knowledge_chunks(index, source_index=index_path)
    target_dir = out_dir or index_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        target_dir / "knowledge_chunks.json",
        {
            "total_chunks": len(chunks),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        },
    )
    write_jsonl(target_dir / "knowledge_chunks.jsonl", chunks)
    (target_dir / "knowledge_chunks.md").write_text(
        render_knowledge_chunks(chunks),
        encoding="utf-8",
    )
    return chunks


def write_jsonl(path: Path, chunks: list[KnowledgeChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, default=str)
        for chunk in chunks
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def render_knowledge_chunks(chunks: list[KnowledgeChunk]) -> str:
    lines = [
        "# Knowledge Chunks",
        "",
        "Embedding-ready chunks for RAG, file search, or long-term review. "
        "Each chunk preserves source URL, topics, scores, and evidence.",
        "",
        f"Total chunks: {len(chunks)}",
        "",
    ]
    if not chunks:
        lines.append("No chunks available.")
        return "\n".join(lines).strip() + "\n"

    for chunk in chunks:
        topics = ", ".join(topic.value for topic in chunk.topics)
        queries = " | ".join(chunk.retrieval_queries[:4])
        lines.extend(
            [
                f"## {chunk.chunk_id}",
                "",
                f"- Title: {chunk.card_title}",
                f"- Topics: {topics}",
                f"- Priority: {chunk.priority_score:.3f}",
                f"- Source: {chunk.source_url}",
                f"- Retrieval queries: {queries}",
                "",
                chunk.text,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
