from datetime import UTC, datetime

from agent_knowledge_harvester.memory.llm_report import build_human_report_user_prompt
from agent_knowledge_harvester.schemas.analysis import KnowledgeTopic
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
    assert "about 8 learning themes" in prompt
    assert "18000-28000" in prompt
    assert "glossary" in prompt.lower()
    assert "source-reported" in prompt
    assert "source names and URLs" in prompt
