from datetime import UTC, datetime

from agent_knowledge_harvester.memory.llm_synthesis import parse_compact_memory_payload
from agent_knowledge_harvester.schemas.memory import AgentMemoryEntry, AgentMemoryPack


def test_parse_compact_memory_payload_falls_back_for_missing_sections() -> None:
    pack = AgentMemoryPack(
        entries=[
            AgentMemoryEntry(
                source_url="https://example.com/mcp",
                card_title="MCP tools",
                claim="MCP standardizes tool access.",
                agent_move="Use explicit MCP server contracts.",
                priority_score=0.8,
                generated_at=datetime(2026, 5, 10, tzinfo=UTC),
            )
        ]
    )

    compact = parse_compact_memory_payload({"core_rules": ["Keep memory short."]}, pack)

    assert compact.core_rules == ["Keep memory short."]
    assert compact.current_patterns
    assert compact.budget == "llm_compact"
