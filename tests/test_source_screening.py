from datetime import UTC, datetime, timedelta

import pytest

from agent_knowledge_harvester.analysis.source_screening import (
    GitHubRepoMetadataClient,
    authority_score_for_domain,
    build_existing_memory_texts,
    distinctive_technical_terms,
    infer_discovery_source_type,
    jaccard_similarity,
    parse_github_repo,
    parse_llm_screening_judgment,
    refine_screening_with_llm,
    score_authority,
    score_candidate,
    score_freshness,
    screen_candidates,
)
from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)
from agent_knowledge_harvester.schemas.ingestion import UrlTarget
from agent_knowledge_harvester.schemas.screening import ScreeningDecision, SourceCandidate


class FakeLLMResult:
    model = "fake-fast"
    prompt_tokens_estimate = 50
    completion_tokens_estimate = 20

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload


class FakeLLMClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls = 0

    async def chat_json(self, *args: object, **kwargs: object) -> FakeLLMResult:
        payload = self.payloads[self.calls]
        self.calls += 1
        return FakeLLMResult(payload)


def test_parse_github_repo_extracts_owner_and_repo() -> None:
    assert parse_github_repo("https://github.com/langchain-ai/langgraph") == (
        "langchain-ai",
        "langgraph",
    )


def test_authority_score_uses_repo_popularity() -> None:
    low = score_authority(SourceCandidate(url="https://github.com/a/b", title="a/b"))
    high = score_authority(
        SourceCandidate(
            url="https://github.com/a/b",
            title="a/b",
            repo_stars=20_000,
            repo_forks=2_000,
        )
    )

    assert high > low
    assert high > 0.7


def test_authority_domain_boost_supports_official_docs() -> None:
    candidate = SourceCandidate(
        url="https://modelcontextprotocol.io/specification/2026-01-01",
        title="Model Context Protocol Specification",
        summary="Protocol for agents and tools.",
    )

    assert authority_score_for_domain(candidate) >= 0.8
    assert infer_discovery_source_type(candidate) == "official_docs"


def test_authority_override_keeps_core_official_specs() -> None:
    screened = score_candidate(
        SourceCandidate(
            url="https://modelcontextprotocol.io/specification/2025-06-18",
            title="Specification - Model Context Protocol",
        )
    )

    assert screened.decision == ScreeningDecision.ACCEPT
    assert "authority_override_candidate" in screened.reasons


def test_freshness_score_prefers_recent_repositories() -> None:
    assert score_freshness(datetime.now(UTC) - timedelta(days=2)) == 1.0
    assert score_freshness(datetime.now(UTC) - timedelta(days=900)) == 0.18


def test_score_candidate_rejects_low_relevance_sources() -> None:
    screened = score_candidate(
        SourceCandidate(
            url="https://github.com/example/calculator",
            title="example/calculator",
            summary="A small calculator library.",
            repo_stars=100_000,
            pushed_at=datetime.now(UTC),
        )
    )

    assert screened.decision == ScreeningDecision.REJECT
    assert screened.relevance_score < 0.16


def test_relevance_does_not_count_arbitrary_topics_as_self_hits() -> None:
    screened = score_candidate(
        SourceCandidate(
            url="https://github.com/example/browser",
            title="Stealth browser",
            summary="Chromium fingerprint patches for browser automation.",
            repo_stars=10_000,
            pushed_at=datetime.now(UTC),
            topics=["browser", "chromium", "stealth", "automation"],
        )
    )

    assert screened.relevance_score == 0.0
    assert screened.decision == ScreeningDecision.REJECT


def test_supporting_terms_without_core_agent_signal_stay_below_threshold() -> None:
    screened = score_candidate(
        SourceCandidate(
            url="https://github.com/example/browser",
            title="Stealth Playwright workflow",
            summary="Browser automation workflows with reliable tests.",
            repo_stars=10_000,
            pushed_at=datetime.now(UTC),
        )
    )

    assert screened.relevance_score < 0.16
    assert screened.decision == ScreeningDecision.REJECT


def test_agent_topic_without_primary_agent_signal_stays_below_threshold() -> None:
    screened = score_candidate(
        SourceCandidate(
            url="https://github.com/example/browser",
            title="Stealth browser",
            summary="Chromium fingerprint patches for browser automation.",
            repo_stars=10_000,
            pushed_at=datetime.now(UTC),
            topics=["ai-agents", "browser-automation"],
        )
    )

    assert screened.relevance_score < 0.16
    assert screened.decision == ScreeningDecision.REJECT


def test_non_authority_missing_date_is_out_of_expanded_scope() -> None:
    screened = score_candidate(
        SourceCandidate(
            url="https://example.com/agent-workflow-post",
            title="Agent workflow post",
            summary="A post about agent workflow orchestration.",
        ),
        min_accept_score=0.1,
        min_relevance_score=0.01,
    )

    assert screened.decision == ScreeningDecision.REJECT
    assert any("recency_status=date_check_required" in reason for reason in screened.reasons)


def test_screen_candidates_accepts_relevant_authoritative_fresh_sources() -> None:
    report = screen_candidates(
        [
            SourceCandidate(
                url="https://github.com/langchain-ai/langgraph",
                title="langchain-ai/langgraph",
                summary="Build durable agent workflows with tool calling and memory.",
                repo_stars=25_000,
                repo_forks=4_000,
                pushed_at=datetime.now(UTC),
                topics=["agents", "workflow"],
            )
        ],
        min_accept_score=0.45,
    )

    assert report.accepted == 1
    assert str(report.selected_urls[0]) == "https://github.com/langchain-ai/langgraph"


def test_novelty_uses_existing_knowledge_index() -> None:
    index = KnowledgeIndex(
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/old",
                card_title="MCP connectors",
                one_sentence="MCP connectors wire agents to tools and workflows.",
                agent_builder_takeaway="Treat MCP as an integration boundary.",
                topics=[KnowledgeTopic.MCP],
                relevance_score=0.8,
                frontier_score=0.5,
                priority_score=0.7,
            )
        ]
    )

    screened = score_candidate(
        SourceCandidate(
            url="https://github.com/example/mcp-connectors",
            title="MCP connectors",
            summary="MCP connectors wire agents to tools and workflows.",
            repo_stars=20_000,
            pushed_at=datetime.now(UTC),
        ),
        existing_texts=build_existing_memory_texts(index),
    )

    assert screened.novelty_score < 0.5
    assert screened.decision == ScreeningDecision.REJECT
    assert "similar_to_existing_memory" in screened.reasons
    assert "possible_better_rewrite_candidate" in screened.reasons


def test_novelty_preserves_new_technical_route_despite_shared_agent_terms() -> None:
    index = KnowledgeIndex(
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/old",
                card_title="MCP connectors",
                one_sentence="MCP connectors wire agents to tools and workflows.",
                agent_builder_takeaway="Treat MCP as an integration boundary.",
                topics=[KnowledgeTopic.MCP],
                relevance_score=0.8,
                frontier_score=0.5,
                priority_score=0.7,
            )
        ]
    )

    screened = score_candidate(
        SourceCandidate(
            url="https://github.com/example/mcp-eval-harness",
            title="MCP evaluation harness",
            summary=(
                "MCP evaluation harness scores tool failures with trace replay "
                "and validation gates."
            ),
            repo_stars=20_000,
            repo_forks=2_000,
            pushed_at=datetime.now(UTC),
        ),
        existing_texts=build_existing_memory_texts(index),
        min_accept_score=0.45,
        min_relevance_score=0.12,
    )

    assert screened.novelty_score >= 0.55
    assert screened.duplicate_of is None
    assert screened.decision == ScreeningDecision.ACCEPT


def test_distinctive_technical_terms_excludes_domain_common_terms() -> None:
    terms = distinctive_technical_terms(
        "MCP agents use workflows, but trace replay and validation harnesses are new."
    )

    assert "mcp" not in terms
    assert "agents" not in terms
    assert "workflows" not in terms
    assert {"trace", "replay", "validation", "harnesses"} <= terms


def test_jaccard_similarity_detects_related_text() -> None:
    assert (
        jaccard_similarity(
            "mcp connectors wire agents to tools",
            "mcp connectors wire agents to tools and workflows",
        )
        > 0.7
    )


def test_parse_llm_screening_judgment_normalizes_bad_values() -> None:
    judgment = parse_llm_screening_judgment(
        {
            "decision": "maybe",
            "agent_relevance": 2,
            "reliability": -1,
            "novelty": "0.4",
            "classic_value": None,
            "memory_action": "save",
            "concerns": ["weak summary"],
        }
    )

    assert judgment.decision == ScreeningDecision.REVIEW
    assert judgment.agent_relevance == 1.0
    assert judgment.reliability == 0.0
    assert judgment.novelty == 0.4
    assert judgment.memory_action == "skip"


@pytest.mark.asyncio
async def test_refine_screening_with_llm_updates_final_decision() -> None:
    report = screen_candidates(
        [
            SourceCandidate(
                url="https://github.com/example/app-agent",
                title="example/app-agent",
                summary=(
                    "A trading app that mentions agents but offers no reusable "
                    "agent framework."
                ),
                repo_stars=8_000,
                repo_forks=500,
                pushed_at=datetime.now(UTC),
                topics=["agents"],
            )
        ],
        min_accept_score=0.4,
    )
    assert report.sources[0].decision == ScreeningDecision.ACCEPT

    refined = await refine_screening_with_llm(
        report,
        FakeLLMClient(
            [
                {
                    "decision": "reject",
                    "agent_relevance": 0.2,
                    "reliability": 0.7,
                    "novelty": 0.3,
                    "classic_value": 0.1,
                    "memory_action": "skip",
                    "reasoning": "Application demo, not reusable agent engineering knowledge.",
                    "concerns": ["application-only"],
                }
            ]
        ),
        max_candidates=1,
    )

    source = refined.sources[0]
    assert refined.llm_enabled is True
    assert refined.llm_judged == 1
    assert refined.rejected == 1
    assert refined.accepted == 0
    assert source.pre_llm_decision == ScreeningDecision.ACCEPT
    assert source.decision == ScreeningDecision.REJECT
    assert source.llm_judgment is not None
    assert source.llm_judgment.reasoning.startswith("Application demo")


@pytest.mark.asyncio
async def test_github_metadata_enrichment_falls_back_on_non_github_url() -> None:
    candidate = await GitHubRepoMetadataClient(Settings()).enrich(
        UrlTarget(url="https://example.com/project")
    )

    assert candidate.title == "https://example.com/project"
