import asyncio

from agent_knowledge_harvester.discovery.query_expansion import build_query_plan
from agent_knowledge_harvester.discovery.search import (
    deduplicate_results,
    normalize_result_url,
    run_search_discovery,
    select_queries,
    write_search_discovery_report,
)
from agent_knowledge_harvester.schemas.analysis import KnowledgeTopic
from agent_knowledge_harvester.schemas.discovery import SearchResult


class FakeSearchProvider:
    name = "fake"

    async def search(
        self,
        query: str,
        *,
        limit: int,
        year: int | None = None,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                url=f"https://example.com/{query.replace(' ', '-')}/?utm_source=test",
                title=f"Result for {query}",
                snippet=f"Snippet for {query} in {year}",
                source=self.name,
                query=query,
                rank=1,
            ),
            SearchResult(
                url="https://example.com/shared?utm_campaign=test#section",
                title="Shared result",
                snippet="Duplicate across queries",
                source=self.name,
                query=query,
                rank=2,
            ),
        ][:limit]


def test_normalize_result_url_removes_tracking_and_fragments() -> None:
    assert (
        normalize_result_url("HTTPS://Example.COM/path/?utm_source=x&a=1#section")
        == "https://example.com/path?a=1"
    )


def test_deduplicate_results_keeps_first_normalized_url() -> None:
    results = [
        SearchResult(url="https://example.com/a?utm_source=x", source="fake", query="q1", rank=1),
        SearchResult(url="https://example.com/a", source="fake", query="q2", rank=1),
        SearchResult(url="https://example.com/b", source="fake", query="q2", rank=2),
    ]

    unique, duplicates = deduplicate_results(results)

    assert [item.url for item in unique] == ["https://example.com/a", "https://example.com/b"]
    assert duplicates == 1


def test_select_queries_prioritizes_scout_and_source_hubs() -> None:
    plan = build_query_plan(topics=[KnowledgeTopic.MEMORY], year=2026)
    queries = select_queries(plan, max_queries=5)

    assert queries[0] == "frontier agent engineering 2026 new pattern"
    assert "AI agent systems 2026 emerging architecture" in queries


def test_run_search_discovery_writes_deduplicated_artifacts(tmp_path) -> None:
    plan = build_query_plan(topics=[KnowledgeTopic.MEMORY], year=2026)

    report = asyncio.run(
        run_search_discovery(
            plan,
            FakeSearchProvider(),
            year=2026,
            max_queries=2,
            results_per_query=2,
            concurrency=2,
        )
    )
    write_search_discovery_report(report, tmp_path)

    assert report.stats.raw_results == 4
    assert report.stats.unique_urls == 3
    assert report.stats.duplicate_results == 1
    assert (tmp_path / "candidate_urls.txt").exists()
    assert (tmp_path / "search_results.md").read_text(encoding="utf-8").startswith(
        "# Search Discovery Results"
    )
