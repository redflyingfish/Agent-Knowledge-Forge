import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.discovery.query_expansion import write_query_plan
from agent_knowledge_harvester.schemas.discovery import (
    DiscoveryStats,
    SearchDiscoveryReport,
    SearchQueryPlan,
    SearchResult,
)
from agent_knowledge_harvester.utils.files import write_json

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_NAMES = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref"}


class SearchProvider(Protocol):
    """Provider-neutral async search interface."""

    name: str

    async def search(
        self,
        query: str,
        *,
        limit: int,
        year: int | None = None,
    ) -> list[SearchResult]:
        """Return normalized search results for one query."""


class SearchProviderNotConfiguredError(RuntimeError):
    """Raised when a requested search provider is missing its API key."""


class SearchAPIError(RuntimeError):
    """Raised when a provider returns an unrecoverable API response."""


class TavilySearchProvider:
    """Tavily search provider."""

    name = "tavily"

    def __init__(self, api_key: str, settings: Settings, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.settings = settings
        self.base_url = base_url or "https://api.tavily.com/search"

    async def search(
        self,
        query: str,
        *,
        limit: int,
        year: int | None = None,
    ) -> list[SearchResult]:
        payload: dict[str, object] = {
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        if year:
            payload["topic"] = "general"
            payload["start_date"] = f"{year}-01-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.settings.user_agent,
        }
        async with httpx.AsyncClient(timeout=self.settings.search_timeout_seconds) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
        ensure_success(response, self.name)
        items = response.json().get("results") or []
        return [
            SearchResult(
                url=str(item.get("url") or ""),
                title=str(item.get("title") or ""),
                snippet=str(item.get("content") or ""),
                source=self.name,
                query=query,
                rank=index,
                raw=compact_raw(item),
            )
            for index, item in enumerate(items, start=1)
            if item.get("url")
        ]


class BraveSearchProvider:
    """Brave Web Search provider."""

    name = "brave"

    def __init__(self, api_key: str, settings: Settings, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.settings = settings
        self.base_url = base_url or "https://api.search.brave.com/res/v1/web/search"

    async def search(
        self,
        query: str,
        *,
        limit: int,
        year: int | None = None,
    ) -> list[SearchResult]:
        params: dict[str, object] = {
            "q": query,
            "count": limit,
            "safesearch": "moderate",
        }
        if year:
            params["freshness"] = f"{year}-01-01to"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
            "User-Agent": self.settings.user_agent,
        }
        async with httpx.AsyncClient(timeout=self.settings.search_timeout_seconds) as client:
            response = await client.get(self.base_url, params=params, headers=headers)
        ensure_success(response, self.name)
        items = (response.json().get("web") or {}).get("results") or []
        return [
            SearchResult(
                url=str(item.get("url") or ""),
                title=str(item.get("title") or ""),
                snippet=str(item.get("description") or ""),
                source=self.name,
                query=query,
                rank=index,
                published_at=item.get("age"),
                language=item.get("language"),
                raw=compact_raw(item),
            )
            for index, item in enumerate(items, start=1)
            if item.get("url")
        ]


class SerpApiSearchProvider:
    """SerpAPI Google Search provider."""

    name = "serpapi"

    def __init__(self, api_key: str, settings: Settings, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.settings = settings
        self.base_url = base_url or "https://serpapi.com/search.json"

    async def search(
        self,
        query: str,
        *,
        limit: int,
        year: int | None = None,
    ) -> list[SearchResult]:
        params: dict[str, object] = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": limit,
        }
        if year:
            params["tbs"] = f"cdr:1,cd_min:1/1/{year}"
        async with httpx.AsyncClient(timeout=self.settings.search_timeout_seconds) as client:
            response = await client.get(self.base_url, params=params)
        ensure_success(response, self.name)
        items = response.json().get("organic_results") or []
        return [
            SearchResult(
                url=str(item.get("link") or ""),
                title=str(item.get("title") or ""),
                snippet=str(item.get("snippet") or ""),
                source=self.name,
                query=query,
                rank=index,
                published_at=item.get("date"),
                raw=compact_raw(item),
            )
            for index, item in enumerate(items[:limit], start=1)
            if item.get("link")
        ]


class ExaSearchProvider:
    """Exa neural/web search provider."""

    name = "exa"

    def __init__(self, api_key: str, settings: Settings, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.settings = settings
        self.base_url = base_url or "https://api.exa.ai/search"

    async def search(
        self,
        query: str,
        *,
        limit: int,
        year: int | None = None,
    ) -> list[SearchResult]:
        payload: dict[str, object] = {
            "query": query,
            "numResults": limit,
            "type": "auto",
            "contents": {"text": False, "highlights": True},
        }
        if year:
            payload["startPublishedDate"] = f"{year}-01-01T00:00:00.000Z"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "User-Agent": self.settings.user_agent,
        }
        async with httpx.AsyncClient(timeout=self.settings.search_timeout_seconds) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
        ensure_success(response, self.name)
        items = response.json().get("results") or []
        return [
            SearchResult(
                url=str(item.get("url") or ""),
                title=str(item.get("title") or ""),
                snippet=snippet_from_exa(item),
                source=self.name,
                query=query,
                rank=index,
                published_at=item.get("publishedDate"),
                raw=compact_raw(item),
            )
            for index, item in enumerate(items, start=1)
            if item.get("url")
        ]


async def run_search_discovery(
    plan: SearchQueryPlan,
    provider: SearchProvider,
    *,
    year: int,
    max_queries: int,
    results_per_query: int,
    concurrency: int,
) -> SearchDiscoveryReport:
    """Execute a query plan and return normalized, deduplicated URL candidates."""
    queries = select_queries(plan, max_queries=max_queries)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    failed_queries: dict[str, str] = {}

    async def run_query(query: str) -> list[SearchResult]:
        async with semaphore:
            try:
                return await search_with_retries(
                    provider,
                    query,
                    limit=results_per_query,
                    year=year,
                )
            except Exception as exc:  # noqa: BLE001 - discovery should report partial failures.
                failed_queries[query] = f"{type(exc).__name__}: {str(exc)[:240]}"
                return []

    batches = await asyncio.gather(*(run_query(query) for query in queries))
    raw_results = [result for batch in batches for result in batch]
    deduped_results, duplicate_results = deduplicate_results(raw_results)
    candidate_urls = [result.url for result in deduped_results]
    stats = DiscoveryStats(
        provider=provider.name,
        planned_queries=count_plan_queries(plan),
        executed_queries=len(queries),
        requested_results_per_query=results_per_query,
        raw_results=len(raw_results),
        unique_urls=len(candidate_urls),
        duplicate_results=duplicate_results,
        failed_queries=len(failed_queries),
        year=year,
    )
    return SearchDiscoveryReport(
        query_plan=plan,
        stats=stats,
        results=deduped_results,
        candidate_urls=candidate_urls,
        failed_queries=failed_queries,
    )


def create_search_provider(
    settings: Settings,
    provider_name: str | None = None,
) -> SearchProvider:
    """Create the configured search provider."""
    name = (provider_name or settings.search_provider or "").strip().lower()
    if not name:
        raise SearchProviderNotConfiguredError(
            "Set AKH_SEARCH_PROVIDER or pass --provider. Supported: tavily, brave, serpapi, exa."
        )

    api_key = search_api_key_for(settings, name)
    if not api_key:
        raise SearchProviderNotConfiguredError(
            f"Missing API key for {name}. Set AKH_SEARCH_API_KEY or AKH_{name.upper()}_API_KEY."
        )
    base_url = str(settings.search_base_url) if settings.search_base_url else None
    if name == "tavily":
        return TavilySearchProvider(api_key, settings, base_url=base_url)
    if name == "brave":
        return BraveSearchProvider(api_key, settings, base_url=base_url)
    if name == "serpapi":
        return SerpApiSearchProvider(api_key, settings, base_url=base_url)
    if name == "exa":
        return ExaSearchProvider(api_key, settings, base_url=base_url)
    raise SearchProviderNotConfiguredError(
        f"Unsupported search provider: {name}. Supported: tavily, brave, serpapi, exa."
    )


def write_search_discovery_report(report: SearchDiscoveryReport, out_dir: Path) -> None:
    """Write search discovery artifacts for review and downstream ingestion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_query_plan(report.query_plan, out_dir)
    write_json(out_dir / "search_results.json", report.model_dump(mode="json"))
    write_json(out_dir / "discovery_stats.json", report.stats.model_dump(mode="json"))
    (out_dir / "candidate_urls.txt").write_text(
        "\n".join(report.candidate_urls) + ("\n" if report.candidate_urls else ""),
        encoding="utf-8",
    )
    (out_dir / "search_results.md").write_text(render_search_results(report), encoding="utf-8")


def render_search_results(report: SearchDiscoveryReport) -> str:
    lines = [
        "# Search Discovery Results",
        "",
        f"- Provider: {report.stats.provider}",
        f"- Year: {report.stats.year}",
        f"- Executed queries: {report.stats.executed_queries}/{report.stats.planned_queries}",
        f"- Raw results: {report.stats.raw_results}",
        f"- Unique URLs: {report.stats.unique_urls}",
        f"- Duplicate results: {report.stats.duplicate_results}",
        f"- Failed queries: {report.stats.failed_queries}",
        "",
        "## Candidate URLs",
        "",
    ]
    lines.extend(f"{index}. {url}" for index, url in enumerate(report.candidate_urls, start=1))
    lines.extend(["", "## Results", ""])
    for result in report.results:
        lines.extend(
            [
                f"### {result.title or result.url}",
                "",
                f"- URL: {result.url}",
                f"- Query: {result.query}",
                f"- Provider rank: {result.rank}",
                f"- Published: {result.published_at or '(unknown)'}",
                f"- Snippet: {trim(result.snippet, 320) or '(none)'}",
                "",
            ]
        )
    if report.failed_queries:
        lines.extend(["## Failed Queries", ""])
        lines.extend(f"- `{query}`: {error}" for query, error in report.failed_queries.items())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def select_queries(plan: SearchQueryPlan, max_queries: int) -> list[str]:
    """Select a balanced query subset from broad scouting, hubs, and topic expansions."""
    if max_queries <= 0:
        return []

    ordered: list[str] = []
    ordered.extend(plan.frontier_scout_queries)
    ordered.extend(plan.source_hub_queries)

    topic_groups: list[list[str]] = []
    for expansion in plan.topic_expansions:
        topic_groups.append(expansion.authority_queries)
        topic_groups.append(expansion.implementation_queries)
        topic_groups.append(expansion.risk_queries)
    ordered.extend(round_robin(topic_groups))
    ordered.extend(plan.stop_signal_queries)
    return unique_strings(ordered)[:max_queries]


def round_robin(groups: Iterable[list[str]]) -> list[str]:
    output: list[str] = []
    remaining = [list(group) for group in groups if group]
    while remaining:
        next_remaining: list[list[str]] = []
        for group in remaining:
            output.append(group.pop(0))
            if group:
                next_remaining.append(group)
        remaining = next_remaining
    return output


def count_plan_queries(plan: SearchQueryPlan) -> int:
    values = list(plan.frontier_scout_queries) + list(plan.source_hub_queries)
    values.extend(plan.stop_signal_queries)
    for expansion in plan.topic_expansions:
        values.extend(expansion.authority_queries)
        values.extend(expansion.implementation_queries)
        values.extend(expansion.risk_queries)
    return len(unique_strings(values))


async def search_with_retries(
    provider: SearchProvider,
    query: str,
    *,
    limit: int,
    year: int,
    attempts: int = 3,
) -> list[SearchResult]:
    """Retry transient search failures with small exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await provider.search(query, limit=limit, year=year)
        except (httpx.TimeoutException, httpx.TransportError, SearchAPIError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(0.6 * (2**attempt))
    if last_error:
        raise last_error
    return []


def deduplicate_results(results: list[SearchResult]) -> tuple[list[SearchResult], int]:
    seen: set[str] = set()
    output: list[SearchResult] = []
    duplicates = 0
    for result in results:
        normalized = normalize_result_url(result.url)
        if not normalized:
            continue
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
        output.append(result.model_copy(update={"url": normalized}))
    return output, duplicates


def normalize_result_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in TRACKING_QUERY_NAMES
        and not any(key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    normalized_query = urlencode(query_items, doseq=True)
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, normalized_query, ""))


def search_api_key_for(settings: Settings, provider_name: str) -> str | None:
    if settings.search_api_key:
        return settings.search_api_key
    provider_keys = {
        "tavily": settings.tavily_api_key,
        "brave": settings.brave_search_api_key,
        "serpapi": settings.serpapi_api_key,
        "exa": settings.exa_api_key,
    }
    return provider_keys.get(provider_name)


def ensure_success(response: httpx.Response, provider_name: str) -> None:
    if response.status_code < 400:
        return
    raise SearchAPIError(
        f"{provider_name} search failed: status={response.status_code} body={response.text[:300]}"
    )


def snippet_from_exa(item: dict[str, object]) -> str:
    highlights = item.get("highlights")
    if isinstance(highlights, list) and highlights:
        return " ".join(str(value) for value in highlights[:3])
    text = item.get("text")
    return str(text or "")


def compact_raw(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        return {}
    allowed = {
        "score",
        "publishedDate",
        "published_at",
        "date",
        "age",
        "language",
        "profile",
        "displayed_link",
    }
    return {key: value for key, value in item.items() if key in allowed}


def unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def trim(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
