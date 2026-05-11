from datetime import date

from agent_knowledge_harvester.analysis.discovery_policy import evaluate_discovery_scope


def test_discovery_scope_accepts_2026_sources_by_default() -> None:
    decision = evaluate_discovery_scope("paper", date(2026, 3, 1))

    assert decision.in_scope is True
    assert decision.status == "in_scope_2026"


def test_discovery_scope_allows_recent_authority_exception() -> None:
    decision = evaluate_discovery_scope("official_docs", date(2025, 6, 1))

    assert decision.in_scope is True
    assert decision.status == "in_scope_authority_exception"


def test_discovery_scope_rejects_old_non_authority_sources() -> None:
    decision = evaluate_discovery_scope("paper", date(2025, 8, 1))

    assert decision.in_scope is False
    assert decision.status == "out_of_future_search_scope"


def test_discovery_scope_allows_very_hot_recent_non_authority_sources() -> None:
    decision = evaluate_discovery_scope("vendor_blog", date(2025, 8, 1), very_hot=True)

    assert decision.in_scope is True
    assert decision.status == "in_scope_very_hot_exception"


def test_discovery_scope_requires_dates_for_unknown_non_authority_sources() -> None:
    decision = evaluate_discovery_scope("github_repository")

    assert decision.in_scope is False
    assert decision.status == "date_check_required"
