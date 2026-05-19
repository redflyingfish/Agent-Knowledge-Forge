from pathlib import Path

from agent_knowledge_harvester.mcp_server.server import (
    KnowledgeMCPRepository,
    create_knowledge_mcp_server,
)
from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)


def test_mcp_repository_searches_cards_and_reads_artifacts(tmp_path: Path) -> None:
    run_dir = build_mcp_run_dir(tmp_path)
    repository = KnowledgeMCPRepository(run_dir)

    summary = repository.summary()
    results = repository.search(query="memory retrieval", topic="memory", limit=5)
    card = repository.get_card(results[0]["card_id"])

    assert summary["total_cards"] == 2
    assert results[0]["card_id"] == "card_0001"
    assert results[0]["topics"] == ["memory"]
    assert card["implementation_notes"] == ["Use recency decay."]
    assert "compact memory" in repository.read_memory_pack("compact")
    assert "Chinese report" in repository.read_human_report("zh")


def test_mcp_repository_truncates_large_artifacts(tmp_path: Path) -> None:
    run_dir = build_mcp_run_dir(tmp_path)
    repository = KnowledgeMCPRepository(run_dir)

    text = repository.read_human_report("en", max_chars=1000)

    assert "[Truncated at 1000 characters." in text


def test_create_knowledge_mcp_server_registers_tools(tmp_path: Path) -> None:
    run_dir = build_mcp_run_dir(tmp_path)

    server = create_knowledge_mcp_server(run_dir)

    assert server.name == "agent-knowledge-forge"


def build_mcp_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    kb_dir = run_dir / "03_knowledge_base"
    memory_dir = run_dir / "04_memory_packs"
    report_dir = run_dir / "05_human_report"
    kb_dir.mkdir(parents=True)
    memory_dir.mkdir()
    report_dir.mkdir()

    index = KnowledgeIndex(
        total_documents=1,
        total_cards=2,
        topic_counts={KnowledgeTopic.MEMORY: 1, KnowledgeTopic.MCP: 1},
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/memory",
                card_title="Memory retrieval",
                one_sentence="Agents should retrieve only useful memories.",
                why_it_matters="Too much context lowers quality.",
                agent_builder_takeaway="Rank memory by relevance and recency.",
                topics=[KnowledgeTopic.MEMORY],
                implementation_notes=["Use recency decay."],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
                evidence=["Selective retrieval keeps prompts small."],
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/mcp",
                card_title="MCP tool server",
                one_sentence="Expose tools with a standard protocol.",
                why_it_matters="Protocol contracts improve portability.",
                agent_builder_takeaway="Expose read-only tools first.",
                topics=[KnowledgeTopic.MCP],
                relevance_score=0.8,
                frontier_score=0.7,
                priority_score=0.75,
            ),
        ],
    )
    (kb_dir / "knowledge_index.json").write_text(index.model_dump_json(), encoding="utf-8")
    (memory_dir / "agent_memory_pack.compact.md").write_text("compact memory", encoding="utf-8")
    (memory_dir / "agent_memory_pack.md").write_text("working memory", encoding="utf-8")
    (memory_dir / "agent_memory_pack.ultra_compact.md").write_text(
        "ultra compact memory",
        encoding="utf-8",
    )
    (memory_dir / "agent_memory_pack.uncompressed.md").write_text(
        "uncompressed memory",
        encoding="utf-8",
    )
    (report_dir / "frontier_learning_report.zh.md").write_text("Chinese report", encoding="utf-8")
    (report_dir / "frontier_learning_report.en.md").write_text(
        "English report " + ("x" * 2000),
        encoding="utf-8",
    )
    (report_dir / "frontier_learning_report.md").write_text("default report", encoding="utf-8")
    return run_dir
