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
    assert "technical blog" in rendered
    assert "中文博客" in rendered
    assert "智能体记忆" in rendered
    assert "frontier agent engineering 2026 new pattern" in rendered
    assert "大模型 Agent 工程实践 2026" in rendered


def test_default_query_plan_covers_broader_agent_engineering_topics() -> None:
    plan = build_query_plan(year=2025)
    topics = {item.topic for item in plan.topic_expansions}

    assert "state_runtime" in topics
    assert "human_in_loop" in topics
    assert "identity_access" in topics
    assert "tool_routing" in topics
    assert "cost_latency" in topics


def test_query_plan_includes_chinese_source_hubs() -> None:
    rendered = render_query_plan(build_query_plan(year=2026))

    assert "site:zhihu.com AI Agent 大模型智能体 2026" in rendered
    assert "site:juejin.cn AI Agent 大模型 工程实践 2026" in rendered
    assert "site:mp.weixin.qq.com AI Agent MCP RAG 智能体 2026" in rendered


def test_parse_topic_values_rejects_unknown_topics() -> None:
    try:
        parse_topic_values(["not_a_topic"])
    except ValueError as exc:
        assert "unknown topic" in str(exc)
    else:
        raise AssertionError("expected ValueError")
