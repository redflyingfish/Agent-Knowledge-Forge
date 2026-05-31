from agent_knowledge_harvester.analysis.topic_discovery import (
    TopicDiscoveryReport,
    bucket_summary,
    direct_known_topic_match,
    discover_topics,
    extract_candidate_topics_from_text,
    recommend_stop_from_batches,
)


def test_extract_candidate_topics_from_text_finds_frontier_hints() -> None:
    observations = extract_candidate_topics_from_text(
        """
        # Memory decay evaluation for long-running agents

        Recent memory decay evaluation for long-running agents is becoming
        important because stale memories can pollute context.
        """,
        source_url="https://example.com/frontier",
        source_bucket="frontier_scout",
    )

    assert observations
    assert observations[0].normalized_topic == "memory decay evaluation long-running agents"
    assert observations[0].source_bucket == "frontier_scout"


def test_direct_known_topic_match_covers_skill_and_tooling_topics() -> None:
    assert direct_known_topic_match("agent skills") == "skills"
    assert direct_known_topic_match("Claude Code") == "claude_code"
    assert direct_known_topic_match("OpenClaw") == "openclaw"


def test_discover_topics_promotes_cross_source_new_topics() -> None:
    observations = []
    observations.extend(
        extract_candidate_topics_from_text(
            "# Memory decay evaluation for long-running agents",
            source_url="https://example.com/a",
            source_bucket="frontier_scout",
        )
    )
    observations.extend(
        extract_candidate_topics_from_text(
            "- Memory decay evaluation for long-running agents",
            source_url="https://example.com/b",
            source_bucket="stop_signal",
        )
    )

    report = discover_topics(
        observations,
        known_topics=["memory", "evaluation", "tool_use"],
        covered_known_topics=["memory", "evaluation"],
        min_sources=2,
        min_confidence=0.45,
    )

    promoted = [candidate for candidate in report.candidates if candidate.promoted]
    assert promoted
    assert promoted[0].topic == "Memory decay evaluation for long-running agents"
    assert promoted[0].is_new
    assert "frontier_scout" in promoted[0].source_buckets
    assert report.metrics.promoted_topic_count == 1
    assert report.metrics.known_topic_coverage == 0.667
    assert bucket_summary(report)["frontier_scout"]["promoted_count"] == 1
    assert bucket_summary(report)["stop_signal"]["promoted_count"] == 1


def test_recommend_stop_from_batches_requires_repeated_low_yield_and_coverage() -> None:
    report = discover_topics(
        [],
        known_topics=["memory", "evaluation", "tool_use"],
        covered_known_topics=["memory", "evaluation", "tool_use"],
    )

    final = recommend_stop_from_batches(
        [report, report, report],
        low_yield_batches=3,
        min_known_topic_coverage=0.65,
    )

    assert isinstance(final, TopicDiscoveryReport)
    assert final.stop_recommendation
    assert "3 consecutive batches" in final.stop_reason
