from datetime import UTC, datetime

from agent_knowledge_harvester.memory.llm_report import (
    build_human_report_user_prompt,
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

    assert "Full Knowledge Card Appendix" in appendix
    assert "A1. Memory retrieval" in appendix
    assert "A2. Tool output offloading" in appendix
    assert "https://example.com/a" in appendix
    assert "Total knowledge cards: 2" in appendix
