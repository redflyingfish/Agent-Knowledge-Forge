from agent_knowledge_harvester.memory.knowledge_clusters import (
    build_knowledge_clusters,
    render_knowledge_clusters,
)
from agent_knowledge_harvester.memory.retrieval_manifest import (
    build_retrieval_manifest,
    render_retrieval_manifest,
    write_retrieval_manifest,
)
from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)


def test_retrieval_manifest_builds_topic_and_source_indexes(tmp_path) -> None:
    index = KnowledgeIndex(
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/mcp",
                card_title="MCP tools",
                one_sentence="MCP standardizes tool access.",
                agent_builder_takeaway="Use MCP servers for tools.",
                topics=[KnowledgeTopic.MCP, KnowledgeTopic.TOOL_USE],
                relevance_score=0.9,
                frontier_score=0.7,
                priority_score=0.8,
            )
        ]
    )

    manifest = build_retrieval_manifest(index, tmp_path / "knowledge_index.json")

    assert manifest.total_entries == 1
    assert manifest.entries[0].memory_id in manifest.topic_index["mcp"]
    assert manifest.entries[0].memory_id in manifest.source_index_map["https://example.com/mcp"]
    assert "MCP tools" in render_retrieval_manifest(manifest)


def test_retrieval_manifest_writes_rag_ready_chunks(tmp_path) -> None:
    index = KnowledgeIndex(
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/memory",
                card_title="Traceable memory",
                one_sentence="Agent memory should preserve source evidence.",
                agent_builder_takeaway="Store URL and evidence with memories.",
                topics=[KnowledgeTopic.MEMORY, KnowledgeTopic.RETRIEVAL],
                relevance_score=0.8,
                frontier_score=0.7,
                priority_score=0.75,
                evidence=["Memory is useful when it remains traceable."],
            )
        ]
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    write_retrieval_manifest(index_path)

    jsonl = tmp_path / "knowledge_chunks.jsonl"
    assert jsonl.exists()
    assert "Traceable memory" in jsonl.read_text(encoding="utf-8")
    assert (tmp_path / "knowledge_chunks.md").exists()
    assert (tmp_path / "knowledge_clusters.md").exists()


def test_knowledge_clusters_group_cards_by_primary_topic(tmp_path) -> None:
    index = KnowledgeIndex(
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/memory",
                card_title="Traceable memory",
                one_sentence="Agent memory should preserve source evidence.",
                agent_builder_takeaway="Store URL and evidence with memories.",
                topics=[KnowledgeTopic.MEMORY, KnowledgeTopic.RETRIEVAL],
                relevance_score=0.8,
                frontier_score=0.7,
                priority_score=0.75,
                evidence=["Memory is useful when it remains traceable."],
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/workflow",
                card_title="Workflow gates",
                one_sentence="Agent workflows need validation gates.",
                agent_builder_takeaway="Add validation gates between phases.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.9,
                frontier_score=0.6,
                priority_score=0.8,
            ),
        ]
    )

    clusters = build_knowledge_clusters(index, tmp_path / "knowledge_index.json")
    rendered = render_knowledge_clusters(clusters)

    assert len(clusters) == 2
    assert clusters[0].avg_priority_score >= clusters[1].avg_priority_score
    assert "Memory" in rendered
    assert "Workflow Design" in rendered
