from datetime import UTC, datetime

from agent_knowledge_harvester.memory.llm_report import (
    build_corpus_facts,
    build_human_report_index_entries,
    build_human_report_index_user_prompt,
    build_human_report_planning_prompt,
    build_human_report_user_prompt,
    build_source_dossiers_from_ingestion_dir,
    insert_corpus_snapshot,
    normalize_report_plan,
    render_full_knowledge_appendix,
)
from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)
from agent_knowledge_harvester.schemas.memory import AgentMemoryEntry, AgentMemoryPack


def test_human_report_prompt_demands_substantial_learning_guide() -> None:
    pack = AgentMemoryPack(
        entries=[
            AgentMemoryEntry(
                source_url=f"https://example.com/{index}",
                card_title="Memory isolation",
                claim="Agents should isolate memory by user.",
                agent_move="Pass entity identifiers when retrieving memories.",
                topics=[KnowledgeTopic.MEMORY],
                priority_score=0.9,
                evidence=["Entity filters prevent cross-user memory leakage."],
                generated_at=datetime(2026, 5, 11, tzinfo=UTC),
            )
            for index in range(55)
        ]
    )

    prompt = build_human_report_user_prompt(pack)

    assert "not a compact summary" in prompt
    assert "enterprise/academic investigation report" in prompt
    assert "Executive" in prompt or "executive" in prompt
    assert "Use evidence fields only to verify synthesis" in prompt
    assert "about 10 learning themes" in prompt
    assert "18000-30000" in prompt
    assert "glossary" in prompt.lower()
    assert "comparison tables" in prompt
    assert "image URLs" in prompt
    assert "source-reported" in prompt
    assert "source names and URLs" in prompt


def test_full_knowledge_appendix_covers_all_index_entries() -> None:
    index = KnowledgeIndex(
        total_documents=1,
        total_cards=2,
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/a",
                source_title="Source A",
                card_title="Memory retrieval",
                one_sentence="Retrieve only relevant long-term memories.",
                why_it_matters="Too much memory pollutes context.",
                agent_builder_takeaway="Rank memories by relevance and recency.",
                topics=[KnowledgeTopic.MEMORY, KnowledgeTopic.CONTEXT_ENGINEERING],
                implementation_notes=["Use a vector index.", "Apply recency decay."],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
                evidence=["Selective retrieval keeps the prompt small."],
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/b",
                card_title="Tool output offloading",
                one_sentence="Store large tool outputs as artifacts.",
                agent_builder_takeaway="Return pointers instead of large blobs.",
                topics=[KnowledgeTopic.TOOL_USE],
                relevance_score=0.8,
                frontier_score=0.7,
                priority_score=0.75,
            ),
        ],
    )

    appendix = render_full_knowledge_appendix(index, language="en")

    assert "Full Source Register Appendix" in appendix
    assert "A1. Memory retrieval" in appendix
    assert "A2. Tool output offloading" in appendix
    assert "https://example.com/a" in appendix
    assert "Total knowledge cards: 2" in appendix
    assert "Selective retrieval keeps the prompt small." not in appendix
    assert "Evidence status" not in appendix


def test_human_report_index_payload_exposes_source_media_assets() -> None:
    index = KnowledgeIndex(
        total_documents=1,
        total_cards=1,
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/a",
                card_title="MCP architecture",
                one_sentence="MCP separates clients and servers.",
                agent_builder_takeaway="Expose external systems as typed tools.",
                topics=[KnowledgeTopic.MCP],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
                image_urls=["https://example.com/diagram.png"],
                table_snippets=[
                    "| Layer | Role |\n| --- | --- |\n| Client | Calls tools |"
                ],
            )
        ],
    )

    entries = build_human_report_index_entries(index)

    assert entries[0]["image_urls"] == ["https://example.com/diagram.png"]
    assert "Client | Calls tools" in entries[0]["table_snippets"][0]


def test_corpus_snapshot_uses_deterministic_index_counts() -> None:
    index = KnowledgeIndex(
        total_documents=2,
        total_cards=3,
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/a",
                card_title="Memory retrieval",
                one_sentence="Retrieve memories selectively.",
                agent_builder_takeaway="Use metadata filters.",
                topics=[KnowledgeTopic.MEMORY],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/a",
                card_title="Memory budgets",
                one_sentence="Budget memory context.",
                agent_builder_takeaway="Compact old entries.",
                topics=[KnowledgeTopic.MEMORY, KnowledgeTopic.CONTEXT_ENGINEERING],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
            ),
            KnowledgeIndexEntry(
                source_url="https://example.com/b",
                card_title="Tool schema",
                one_sentence="Use strict tool schemas.",
                agent_builder_takeaway="Validate parameters.",
                topics=[KnowledgeTopic.TOOL_USE],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
            ),
        ],
    )

    facts = build_corpus_facts(index)
    report = insert_corpus_snapshot("# Frontier Agent Development Learning Report\n\nBody", index)

    assert facts["unique_sources"] == 2
    assert facts["topic_count"] == 3
    assert "## Report Snapshot" in report
    assert "| Knowledge cards | 3 |" in report
    assert "| Unique source URLs | 2 |" in report


def test_source_dossiers_add_long_source_context_for_human_report(tmp_path) -> None:
    index = KnowledgeIndex(
        total_documents=1,
        total_cards=1,
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/research",
                source_title="Agent Runtime Paper",
                card_title="Stateful runtime isolates agent sessions",
                one_sentence="Agent runtimes should isolate sessions.",
                agent_builder_takeaway="Use per-session runtime boundaries.",
                topics=[KnowledgeTopic.STATE_RUNTIME, KnowledgeTopic.AGENT_HARDENING],
                relevance_score=0.95,
                frontier_score=0.9,
                priority_score=0.925,
            )
        ],
    )
    ingestion_payload = {
        "success": True,
        "clean": {
            "source_url": "https://example.com/research",
            "title": "Agent Runtime Paper",
            "summary_hint": "A paper about runtime isolation.",
            "markdown": "\n\n".join(
                [
                    "# Agent Runtime Paper",
                    "## Architecture",
                    (
                        "The runtime architecture separates the model planner from the "
                        "execution sandbox. This design limits credential exposure, records "
                        "tool calls, and gives operators a deterministic boundary for "
                        "security review."
                    ),
                    "## Evaluation",
                    (
                        "The evaluation method measures recovery after failed tool calls, "
                        "latency overhead, and whether state is preserved across long "
                        "workflows without leaking data between sessions."
                    ),
                ]
            ),
        },
    }
    (tmp_path / "research.json").write_text(
        __import__("json").dumps(ingestion_payload),
        encoding="utf-8",
    )

    dossiers = build_source_dossiers_from_ingestion_dir(index, tmp_path)
    prompt = build_human_report_index_user_prompt(index, source_dossiers=dossiers)

    assert len(dossiers) == 1
    assert dossiers[0]["source_outline"] == ["Agent Runtime Paper", "Architecture", "Evaluation"]
    assert "execution sandbox" in dossiers[0]["selected_passages"]
    assert "Source dossiers for deep synthesis" in prompt
    assert "Knowledge cards provide the claim index" in prompt


def test_human_report_uses_planning_handoff() -> None:
    index = KnowledgeIndex(
        total_documents=1,
        total_cards=1,
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/security",
                card_title="Runtime guardrails need policy checks",
                one_sentence="Guardrails should be enforced at runtime.",
                agent_builder_takeaway="Validate tool calls before execution.",
                topics=[KnowledgeTopic.GUARDRAILS],
                relevance_score=0.9,
                frontier_score=0.8,
                priority_score=0.85,
            )
        ],
    )

    planning_prompt = build_human_report_planning_prompt(index, language="zh")
    writer_prompt = build_human_report_index_user_prompt(
        index,
        language="zh",
        report_plan="## 计划\n\n- 先解释 runtime guardrails，再比较替代方案。",
    )
    normalized = normalize_report_plan("```markdown\n## Plan\n\nDetailed plan.\n```")

    assert "Create a Markdown 研究报告写作计划" in planning_prompt
    assert "Propose 10-14 major sections" in planning_prompt
    assert "deep treatment" in planning_prompt
    assert "Claude Code, Skills, OpenClaw" in planning_prompt
    assert "Research report plan to follow" in writer_prompt
    assert "先解释 runtime guardrails" in writer_prompt
    assert "depth contract" in writer_prompt
    assert "4-7 coherent paragraphs" in writer_prompt
    assert normalized == "## Plan\n\nDetailed plan."
