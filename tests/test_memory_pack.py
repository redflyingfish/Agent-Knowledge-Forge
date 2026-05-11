from datetime import UTC, datetime

from agent_knowledge_harvester.memory.pack import (
    build_compact_memory_pack,
    build_memory_pack,
    discover_index_paths,
    learning_theme,
    parse_after,
    render_compact_memory_pack,
    render_human_learning_report,
    render_memory_pack,
    write_memory_pack,
    write_uncompressed_memory_pack,
)
from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)


def test_parse_after_accepts_date_and_datetime() -> None:
    assert parse_after("2026-05-09") == datetime(2026, 5, 9, tzinfo=UTC)
    assert parse_after("2026-05-09T12:00:00Z") == datetime(2026, 5, 9, 12, tzinfo=UTC)


def test_build_memory_pack_filters_by_time_priority_and_duplicates(tmp_path) -> None:
    old_index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 1, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/old",
                card_title="Old MCP connector",
                one_sentence="MCP connectors wire agents to tools.",
                agent_builder_takeaway="Treat MCP as an integration boundary.",
                topics=[KnowledgeTopic.MCP],
                relevance_score=0.8,
                frontier_score=0.5,
                priority_score=0.8,
            )
        ],
    )
    new_index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/new",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW, KnowledgeTopic.EVALUATION],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
                evidence=["Agent workflows should use explicit validation gates."],
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/dupe",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW, KnowledgeTopic.EVALUATION],
                relevance_score=0.69,
                frontier_score=0.4,
                priority_score=0.68,
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/low",
                card_title="Small note",
                one_sentence="A small note about agents.",
                agent_builder_takeaway="A small note about agents.",
                topics=[KnowledgeTopic.AGENT_ARCHITECTURE],
                relevance_score=0.2,
                frontier_score=0.1,
                priority_score=0.1,
            ),
        ],
    )
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(old_index.model_dump_json(), encoding="utf-8")
    new_path.write_text(new_index.model_dump_json(), encoding="utf-8")

    pack = build_memory_pack(
        [old_path, new_path],
        after=datetime(2026, 5, 9, tzinfo=UTC),
        min_priority=0.5,
    )

    assert pack.total_input_entries == 4
    assert pack.retained_entries == 1
    assert pack.dropped_by_time == 1
    assert pack.dropped_by_priority == 1
    assert pack.dropped_duplicates == 1
    assert pack.entries[0].card_title == "Workflow gates"


def test_build_memory_pack_can_keep_uncompressed_entries(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/a",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
                evidence=["First source evidence."],
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/b",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.69,
                evidence=["Second source evidence."],
            ),
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    pack = build_memory_pack([index_path], max_entries=None, dedupe_threshold=None)

    assert pack.retained_entries == 2
    assert pack.dropped_duplicates == 0


def test_render_memory_pack_is_compact_and_agent_readable(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/workflow",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
            )
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    rendered = render_memory_pack(build_memory_pack([index_path]))

    assert "# Agent Memory Pack" in rendered
    assert "Agent workflows should use explicit validation gates." in rendered
    assert "Agent move:" in rendered
    assert len(rendered) < 1000


def test_render_memory_pack_hides_evidence_by_default(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/workflow",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
                evidence=["Source evidence should be optional."],
            )
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")
    pack = build_memory_pack([index_path])

    assert "Source evidence should be optional." not in render_memory_pack(pack)
    assert "Source evidence should be optional." in render_memory_pack(
        pack,
        include_evidence=True,
    )


def test_render_human_learning_report_is_english_and_source_attributed(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/workflow",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
                evidence=["Validation gates keep long-running workflows inspectable."],
            )
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    rendered = render_human_learning_report(build_memory_pack([index_path]))

    assert "# Frontier Agent Development Learning Report" in rendered
    assert "**Source:** https://example.com/workflow" in rendered
    assert "## Workflow Design And Validation" in rendered
    assert "**Practice question:**" in rendered
    assert "Validation gates keep long-running workflows inspectable." in rendered


def test_learning_theme_groups_entries_for_readable_report() -> None:
    assert (
        learning_theme(
            _memory_entry_for_topics([KnowledgeTopic.MCP, KnowledgeTopic.TOOL_USE])
        )
        == "Tool And Data Integration"
    )
    assert (
        learning_theme(
            _memory_entry_for_topics([KnowledgeTopic.WORKFLOW, KnowledgeTopic.EVALUATION])
        )
        == "Workflow Design And Validation"
    )


def test_write_memory_pack_writes_agent_and_human_markdown(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/workflow",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
            )
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    out_dir = tmp_path / "memory"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    write_memory_pack(
        build_memory_pack([index_path]),
        out_dir,
        uncompressed_pack=build_memory_pack(
            [index_path],
            max_entries=None,
            dedupe_threshold=None,
        ),
    )

    assert (out_dir / "agent_memory_pack.md").exists()
    assert (out_dir / "agent_memory_pack.uncompressed.md").exists()
    assert (out_dir / "frontier_learning_report.md").exists()
    assert (out_dir / "agent_memory_pack.json").exists()
    assert (out_dir / "agent_memory_pack.compact.md").exists()
    assert (out_dir / "agent_memory_pack.ultra_compact.md").exists()


def test_write_uncompressed_memory_pack_includes_evidence(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/workflow",
                card_title="Workflow gates",
                one_sentence="Agent workflows should use explicit validation gates.",
                agent_builder_takeaway="Model work as phases with validation gates.",
                topics=[KnowledgeTopic.WORKFLOW],
                relevance_score=0.7,
                frontier_score=0.4,
                priority_score=0.7,
                evidence=["Evidence must stay in long-term memory."],
            )
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    out_dir = tmp_path / "memory"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    write_uncompressed_memory_pack(
        build_memory_pack([index_path], max_entries=None, dedupe_threshold=None),
        out_dir,
    )

    rendered = (out_dir / "agent_memory_pack.uncompressed.md").read_text(encoding="utf-8")
    assert "# Agent Memory Pack - Uncompressed Long-Term Layer" in rendered
    assert "Evidence must stay in long-term memory." in rendered


def test_compact_memory_pack_contains_rules_patterns_and_retrieval_pointers(tmp_path) -> None:
    index = KnowledgeIndex(
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/mcp",
                card_title="MCP tools",
                one_sentence="MCP standardizes agent access to tools and data.",
                agent_builder_takeaway="Expose tools through explicit MCP server contracts.",
                topics=[KnowledgeTopic.MCP, KnowledgeTopic.TOOL_USE],
                relevance_score=0.9,
                frontier_score=0.7,
                priority_score=0.8,
            )
        ],
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    compact = build_compact_memory_pack(build_memory_pack([index_path]))
    rendered = render_compact_memory_pack(compact)

    assert "## Core Rules" in rendered
    assert "MCP tools" in rendered
    assert "topic:mcp" in rendered


def test_discover_index_paths_accepts_dirs_and_files(tmp_path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    index_path = analysis_dir / "knowledge_index.json"
    index_path.write_text(KnowledgeIndex().model_dump_json(), encoding="utf-8")

    assert discover_index_paths([analysis_dir, index_path]) == [index_path]


def _memory_entry_for_topics(topics: list[KnowledgeTopic]):
    from agent_knowledge_harvester.schemas.memory import AgentMemoryEntry

    return AgentMemoryEntry(
        source_url="https://example.com/source",
        card_title="Card",
        claim="Claim",
        agent_move="Move",
        topics=topics,
        priority_score=0.5,
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
    )
