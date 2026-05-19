from datetime import date

from agent_knowledge_harvester.analysis.discovery_policy import evaluate_discovery_scope


def test_discovery_scope_accepts_2026_sources_by_default() -> None:
    decision = evaluate_discovery_scope("paper", date(2026, 3, 1))

    assert decision.in_scope is True
    assert decision.status == "in_scope_2025_plus"


def test_discovery_scope_accepts_2025_blogs_by_default() -> None:
    decision = evaluate_discovery_scope("vendor_blog", date(2025, 2, 1))

    assert decision.in_scope is True
    assert decision.status == "in_scope_2025_plus"


def test_discovery_scope_allows_recent_authority_exception() -> None:
    decision = evaluate_discovery_scope("official_docs", date(2025, 6, 1))

    assert decision.in_scope is True
    assert decision.status == "in_scope_2025_plus"


def test_discovery_scope_allows_old_non_authority_sources_with_advisory_status() -> None:
    decision = evaluate_discovery_scope("paper", date(2024, 12, 31))

    assert decision.in_scope is True
    assert decision.status == "legacy_date_allowed"


def test_discovery_scope_allows_very_hot_recent_non_authority_sources() -> None:
    decision = evaluate_discovery_scope("vendor_blog", date(2025, 8, 1), very_hot=True)

    assert decision.in_scope is True
    assert decision.status == "in_scope_2025_plus"


def test_discovery_scope_allows_unknown_dates_for_non_authority_sources() -> None:
    decision = evaluate_discovery_scope("github_repository")

    assert decision.in_scope is True
    assert decision.status == "date_unknown_allowed"
