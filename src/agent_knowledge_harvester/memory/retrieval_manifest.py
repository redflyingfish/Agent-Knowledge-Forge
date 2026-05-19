from pathlib import Path

from agent_knowledge_harvester.memory.knowledge_chunks import write_knowledge_chunks
from agent_knowledge_harvester.memory.knowledge_clusters import write_knowledge_clusters
from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex, KnowledgeIndexEntry
from agent_knowledge_harvester.schemas.memory import RetrievalManifest, RetrievalManifestEntry
from agent_knowledge_harvester.utils.files import stable_slug, write_json


def build_retrieval_manifest(index: KnowledgeIndex, source_index: Path) -> RetrievalManifest:
    """Build a RAG/MCP-friendly manifest from the structured knowledge index."""
    entries = [
        build_manifest_entry(item, position)
        for position, item in enumerate(index.entries, start=1)
    ]
    topic_index: dict[str, list[str]] = {}
    source_index_map: dict[str, list[str]] = {}
    for entry in entries:
        for topic in entry.topics:
            topic_index.setdefault(topic.value, []).append(entry.memory_id)
        source_index_map.setdefault(str(entry.source_url), []).append(entry.memory_id)
    return RetrievalManifest(
        source_index=str(source_index),
        total_entries=len(entries),
        entries=entries,
        topic_index=dict(sorted(topic_index.items())),
        source_index_map=dict(sorted(source_index_map.items())),
    )


def build_manifest_entry(
    item: KnowledgeIndexEntry,
    position: int,
) -> RetrievalManifestEntry:
    slug = stable_slug(
        " ".join([str(item.source_url), item.card_title, item.one_sentence]),
        max_len=48,
    )
    return RetrievalManifestEntry(
        memory_id=f"k{position:04d}-{slug}",
        source_url=item.source_url,
        source_title=item.source_title,
        card_title=item.card_title,
        claim=item.one_sentence,
        agent_move=item.agent_builder_takeaway,
        topics=item.topics,
        priority_score=item.priority_score,
        relevance_score=item.relevance_score,
        frontier_score=item.frontier_score,
        evidence=item.evidence,
    )


def write_retrieval_manifest(index_path: Path, out_dir: Path | None = None) -> RetrievalManifest:
    index = KnowledgeIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    manifest = build_retrieval_manifest(index, source_index=index_path)
    target_dir = out_dir or index_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "retrieval_manifest.json", manifest.model_dump(mode="json"))
    (target_dir / "retrieval_manifest.md").write_text(
        render_retrieval_manifest(manifest),
        encoding="utf-8",
    )
    write_knowledge_chunks(index_path, out_dir=target_dir)
    write_knowledge_clusters(index_path, out_dir=target_dir)
    return manifest


def render_retrieval_manifest(manifest: RetrievalManifest) -> str:
    lines = [
        "# Retrieval Manifest",
        "",
        f"Source index: {manifest.source_index}",
        f"Total entries: {manifest.total_entries}",
        "",
        "## Topic Index",
        "",
    ]
    if manifest.topic_index:
        for topic, ids in manifest.topic_index.items():
            lines.append(f"- {topic}: {', '.join(ids)}")
    else:
        lines.append("No topic entries.")

    lines.extend(["", "## Entries", ""])
    for entry in manifest.entries:
        topics = ", ".join(topic.value for topic in entry.topics)
        lines.extend(
            [
                f"### {entry.memory_id}",
                "",
                f"- Title: {entry.card_title}",
                f"- Topics: {topics}",
                f"- Priority: {entry.priority_score:.3f}",
                f"- Claim: {entry.claim}",
                f"- Agent move: {entry.agent_move}",
                f"- Source: {entry.source_url}",
                f"- Evidence count: {len(entry.evidence)}",
                "",
            ]
        )
    lines.extend(
        [
            "## RAG-Ready Chunk Files",
            "",
            (
                "- `knowledge_chunks.jsonl`: newline-delimited chunks for vector stores "
                "or file search."
            ),
            "- `knowledge_chunks.json`: same chunks as a single JSON document.",
            "- `knowledge_chunks.md`: readable review copy of the chunk layer.",
            "- `knowledge_clusters.md/json`: topic clusters for browsing and gap analysis.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"
