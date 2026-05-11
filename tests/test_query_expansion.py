from agent_knowledge_harvester.discovery.query_expansion import (
    build_query_plan,
    parse_topic_values,
    render_query_plan,
)
from agent_knowledge_harvester.schemas.analysis import KnowledgeTopic


def test_query_plan_expands_memory_rag_and_unknown_scout_queries() -> None:
    plan = build_query_plan(
        topics=[KnowledgeTopic.MEMORY, KnowledgeTopic.RAG, KnowledgeTopic.AGENT_HARDENING],
        year=2026,
    )
    rendered = render_query_plan(plan)

    assert [item.topic for item in plan.topic_expansions] == [
        "memory",
        "rag",
        "agent_hardening",
    ]
    assert "context engineering" in rendered
    assert "agentic RAG" in rendered
    assert "frontier agent engineering 2026 new pattern" in rendered


def test_parse_topic_values_rejects_unknown_topics() -> None:
    try:
        parse_topic_values(["not_a_topic"])
    except ValueError as exc:
        assert "unknown topic" in str(exc)
    else:
        raise AssertionError("expected ValueError")
