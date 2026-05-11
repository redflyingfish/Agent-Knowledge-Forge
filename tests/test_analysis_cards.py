from agent_knowledge_harvester.analysis.knowledge_cards import (
    KnowledgeCardAnalyzer,
    analyze_directory,
    build_frontier_brief,
    build_knowledge_index,
    clamp_text,
    detect_topics,
    is_low_value_section,
    normalize_table_rows,
    render_frontier_brief,
    render_index_markdown,
    split_markdown_sections,
    split_sentences,
    strip_markdown,
    suggest_next_experiments,
    write_frontier_brief_from_index,
)
from agent_knowledge_harvester.schemas.analysis import AnalysisResult, KnowledgeCard, KnowledgeTopic
from agent_knowledge_harvester.schemas.ingestion import (
    CleanDocument,
    FetchMethod,
    FetchStats,
    IngestionResult,
    UrlTarget,
)


def test_split_markdown_sections_groups_heading_bodies() -> None:
    sections = split_markdown_sections(
        """
        Intro paragraph.

        ## What can MCP enable?

        Agents can connect to tools and workflows.

        ## Why does MCP matter?

        MCP reduces integration complexity.
        """
    )

    assert sections[0][0] == ""
    assert sections[1][0] == "What can MCP enable?"
    assert "tools and workflows" in sections[1][1]


def test_detect_topics_finds_agent_mcp_and_tools() -> None:
    topics = detect_topics("MCP lets agents connect to external tools and data sources.")

    assert KnowledgeTopic.MCP in topics
    assert KnowledgeTopic.AGENT_ARCHITECTURE in topics
    assert KnowledgeTopic.TOOL_USE in topics


def test_detect_topics_finds_newer_agent_engineering_axes() -> None:
    topics = detect_topics(
        "Agent hardening needs context engineering, agentic RAG, observability, and evals."
    )

    assert KnowledgeTopic.AGENT_HARDENING in topics
    assert KnowledgeTopic.CONTEXT_ENGINEERING in topics
    assert KnowledgeTopic.RAG in topics
    assert KnowledgeTopic.OBSERVABILITY in topics
    assert KnowledgeTopic.EVALUATION in topics


def test_sentence_splitter_handles_reader_spacing_and_abbreviations() -> None:
    sentences = split_sentences(
        "Agents connect to systems.Using MCP, they can call tools (e.g. search engines)."
        "Think of this as an integration boundary."
    )

    assert sentences == [
        "Agents connect to systems.",
        "Using MCP, they can call tools (e.g. search engines).",
        "Think of this as an integration boundary.",
    ]


def test_strip_markdown_removes_empty_links_and_code_blocks() -> None:
    stripped = strip_markdown("[](https://example.com) ```python\nprint('x')\n``` **MCP**")

    assert "https://example.com" not in stripped
    assert "print" not in stripped
    assert stripped == "MCP"


def test_normalize_table_rows_turns_cells_into_sentences() -> None:
    normalized = normalize_table_rows(
        """
        | What it is | Where it lives |
        | --- | --- |
        | Connectors | MCP servers that wire Claude to your data |
        """
    )

    assert "Connectors: MCP servers that wire Claude to your data." in normalized


def test_sentence_splitter_filters_platform_navigation() -> None:
    sentences = split_sentences(
        "GitHub Navigation Menu Toggle navigation Appearance settings. "
        "MCP servers connect Claude to your data."
    )

    assert sentences == ["MCP servers connect Claude to your data."]


def test_sentence_splitter_filters_file_listing_noise() -> None:
    sentences = split_sentences(
        ".github/workflows: .github/workflows. Last commit message: Last commit date. "
        "MCP connectors wire Claude to terminals and document stores."
    )

    assert sentences == ["MCP connectors wire Claude to terminals and document stores."]


def test_low_value_sections_are_skipped_before_card_scoring() -> None:
    assert is_low_value_section("Prerequisites")
    assert is_low_value_section("Experimental: AI-Assisted Setup (Release Download)")
    assert is_low_value_section("Option 2: AGENTS.md (Simple Alternative)")
    assert is_low_value_section("OpenAI Codex")
    assert not is_low_value_section("Three-Phase Adaptive Workflow")


def test_analyzer_prefers_workflow_content_over_installation_sections() -> None:
    document = CleanDocument(
        source_url="https://example.com/aidlc",
        title="AI-DLC",
        markdown="""
        ## Prerequisites

        Have one of our supported platforms/tools for Assisted AI Coding installed:
        Claude Code CLI: Install. Cursor: Installation Link.

        ## Additional Resources

        Working with AI-DLC: docs/WORKING-WITH-AIDLC.md.
        Contributing Guidelines: CONTRIBUTING.md.

        ## Three-Phase Adaptive Workflow

        AI-DLC guides agents through Inception, Construction, and Operations phases.
        Inception clarifies requirements before coding begins.
        Construction turns requirements into implementation units.
        Operations adds evaluation, validation, quality checks, and workflow readiness.
        """,
        summary_hint="AI-DLC provides adaptive workflow steering for coding agents.",
        technical_signal_score=0.7,
        stats=FetchStats(method=FetchMethod.JINA_READER, source_url="https://example.com/aidlc"),
    )

    result = KnowledgeCardAnalyzer().analyze_document(document)

    assert [card.title for card in result.cards] == ["Three-Phase Adaptive Workflow"]
    assert "workflow" in result.cards[0].one_sentence.lower()


def test_clamp_text_preserves_word_boundaries() -> None:
    assert clamp_text("alpha beta gamma", 12) == "alpha beta..."


def test_analyzer_builds_compact_agent_knowledge_card() -> None:
    document = CleanDocument(
        source_url="https://example.com/mcp",
        title="MCP Introduction",
        markdown="""
        MCP is an open-source standard for connecting AI applications to external systems.

        ## Why does MCP matter?

        Developers can reduce development time when building an AI application or agent.
        Agents can access data sources, tools, and workflows to perform useful tasks.
        The ecosystem is supported by Claude, ChatGPT, Visual Studio Code, and Cursor.
        """,
        summary_hint="MCP connects AI applications to external systems.",
        technical_signal_score=0.7,
        stats=FetchStats(method=FetchMethod.JINA_READER, source_url="https://example.com/mcp"),
    )

    result = KnowledgeCardAnalyzer().analyze_document(document)

    assert result.cards
    card = result.cards[0]
    assert card.relevance_score > 0.2
    assert card.frontier_score > 0.2
    assert KnowledgeTopic.MCP in card.topics
    assert "integration boundary" in card.agent_builder_takeaway


def test_analyzer_skips_sections_without_useful_sentences() -> None:
    document = CleanDocument(
        source_url="https://example.com/noise",
        title="Noise",
        markdown="""
        ## Repository Layout

        GitHub Navigation Menu Toggle navigation Appearance settings.
        """,
        summary_hint="GitHub Navigation Menu Toggle navigation Appearance settings.",
        technical_signal_score=0.2,
        stats=FetchStats(method=FetchMethod.JINA_READER, source_url="https://example.com/noise"),
    )

    result = KnowledgeCardAnalyzer().analyze_document(document)

    assert result.cards == []
    assert result.skipped_reason == "no section passed the relevance threshold"


def test_knowledge_index_prioritizes_frontier_relevant_cards() -> None:
    source_url = "https://example.com/mcp"
    result = AnalysisResult(
        source_url=source_url,
        title="MCP",
        cards=[
            KnowledgeCard(
                source_url=source_url,
                title="Low value",
                one_sentence="A small note.",
                why_it_matters="A small note.",
                agent_builder_takeaway="A small note.",
                topics=[KnowledgeTopic.AGENT_ARCHITECTURE],
                relevance_score=0.2,
                frontier_score=0.1,
            ),
            KnowledgeCard(
                source_url=source_url,
                title="MCP connectors",
                one_sentence="MCP connectors wire agents to tools and workflows.",
                why_it_matters="MCP connectors wire agents to tools and workflows.",
                agent_builder_takeaway="Treat MCP as an integration boundary.",
                topics=[KnowledgeTopic.MCP, KnowledgeTopic.TOOL_USE, KnowledgeTopic.WORKFLOW],
                evidence=["MCP connectors wire agents to tools and workflows."],
                relevance_score=0.8,
                frontier_score=0.6,
            ),
        ],
    )

    index = build_knowledge_index([result])

    assert index.entries[0].card_title == "MCP connectors"
    assert index.topic_counts[KnowledgeTopic.MCP] == 1
    assert index.entries[0].priority_score > index.entries[1].priority_score


def test_render_index_markdown_includes_priority_cards() -> None:
    source_url = "https://example.com/mcp"
    index = build_knowledge_index(
        [
            AnalysisResult(
                source_url=source_url,
                cards=[
                    KnowledgeCard(
                        source_url=source_url,
                        title="MCP connectors",
                        one_sentence="MCP connectors wire agents to tools.",
                        why_it_matters="MCP connectors wire agents to tools.",
                        agent_builder_takeaway="Treat MCP as an integration boundary.",
                        topics=[KnowledgeTopic.MCP],
                        relevance_score=0.7,
                        frontier_score=0.5,
                    )
                ],
            )
        ]
    )

    rendered = render_index_markdown(index)

    assert "# Knowledge Index" in rendered
    assert "Priority:" in rendered
    assert "MCP connectors" in rendered


def test_frontier_brief_summarizes_priority_index() -> None:
    source_url = "https://example.com/mcp"
    index = build_knowledge_index(
        [
            AnalysisResult(
                source_url=source_url,
                cards=[
                    KnowledgeCard(
                        source_url=source_url,
                        title="MCP connectors",
                        one_sentence="MCP connectors wire agents to tools.",
                        why_it_matters="MCP connectors wire agents to tools.",
                        agent_builder_takeaway="Treat MCP as an integration boundary.",
                        topics=[KnowledgeTopic.MCP, KnowledgeTopic.TOOL_USE],
                        relevance_score=0.8,
                        frontier_score=0.6,
                    )
                ],
            )
        ]
    )

    brief = build_frontier_brief(index)
    rendered = render_frontier_brief(brief)

    assert "Top priority is 'MCP connectors'" in brief.summary
    assert brief.top_signals == ["MCP connectors wire agents to tools."]
    assert brief.agent_builder_moves == ["Treat MCP as an integration boundary."]
    assert brief.next_experiments == [
        "Map one target workflow into data sources, tools, and MCP connectors.",
        "Convert one repeated action into a typed tool with clear failure handling.",
    ]
    assert "## Agent Builder Moves" in rendered
    assert "## Next Experiments" in rendered


def test_suggest_next_experiments_uses_topics() -> None:
    source_url = "https://example.com/multi-agent"
    entries = [
        build_knowledge_index(
            [
                AnalysisResult(
                    source_url=source_url,
                    cards=[
                        KnowledgeCard(
                            source_url=source_url,
                            title="Handoffs",
                            one_sentence="Agents hand off work through an orchestration layer.",
                            why_it_matters="Agents hand off work through an orchestration layer.",
                            agent_builder_takeaway="Look for handoff state.",
                            topics=[KnowledgeTopic.MULTI_AGENT, KnowledgeTopic.RETRIEVAL],
                            relevance_score=0.7,
                            frontier_score=0.3,
                        )
                    ],
                )
            ]
        ).entries[0]
    ]

    suggestions = suggest_next_experiments(entries)

    assert suggestions == [
        "Prototype the smallest handoff loop and record what state crosses agents.",
        "Test whether retrieved context preserves enough source evidence for action.",
    ]


def test_analyze_directory_writes_cards_and_priority_index(tmp_path) -> None:
    in_dir = tmp_path / "ingested"
    out_dir = tmp_path / "analysis"
    in_dir.mkdir()
    result = IngestionResult(
        target=UrlTarget(url="https://example.com/mcp"),
        success=True,
        clean=CleanDocument(
            source_url="https://example.com/mcp",
            title="MCP",
            markdown="MCP connectors let agents connect to tools and workflows.",
            summary_hint="MCP connectors let agents connect to tools and workflows.",
            technical_signal_score=0.7,
            stats=FetchStats(
                method=FetchMethod.JINA_READER,
                source_url="https://example.com/mcp",
            ),
        ),
    )
    (in_dir / "mcp.json").write_text(result.model_dump_json(), encoding="utf-8")

    _, stats, index = analyze_directory(in_dir, out_dir)

    assert stats.total_cards == 1
    assert index.entries[0].card_title == "MCP"
    assert (out_dir / "knowledge_index.md").exists()
    assert (out_dir / "knowledge_index.json").exists()
    assert (out_dir / "frontier_brief.md").exists()
    assert (out_dir / "frontier_brief.json").exists()


def test_analyze_directory_skips_non_ingestion_json(tmp_path) -> None:
    in_dir = tmp_path / "analysis"
    out_dir = tmp_path / "analysis-out"
    in_dir.mkdir()
    (in_dir / "knowledge_index.json").write_text('{"entries":[]}', encoding="utf-8")

    _, stats, index = analyze_directory(in_dir, out_dir)

    assert stats.total_documents == 0
    assert index.entries == []
    assert (out_dir / "frontier_brief.md").exists()


def test_write_frontier_brief_from_index(tmp_path) -> None:
    source_url = "https://example.com/mcp"
    index = build_knowledge_index(
        [
            AnalysisResult(
                source_url=source_url,
                cards=[
                    KnowledgeCard(
                        source_url=source_url,
                        title="MCP connectors",
                        one_sentence="MCP connectors wire agents to tools.",
                        why_it_matters="MCP connectors wire agents to tools.",
                        agent_builder_takeaway="Treat MCP as an integration boundary.",
                        topics=[KnowledgeTopic.MCP],
                        relevance_score=0.7,
                        frontier_score=0.5,
                    )
                ],
            )
        ]
    )
    index_path = tmp_path / "knowledge_index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    brief = write_frontier_brief_from_index(index_path)

    assert brief.top_signals == ["MCP connectors wire agents to tools."]
    assert (tmp_path / "frontier_brief.md").exists()
