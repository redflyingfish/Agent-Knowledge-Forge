from agent_knowledge_harvester.memory.retrieval_manifest import (
    build_retrieval_manifest,
    render_retrieval_manifest,
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
