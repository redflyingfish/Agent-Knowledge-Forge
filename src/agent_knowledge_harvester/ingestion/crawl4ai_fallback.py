import logging
import time

from markdownify import markdownify as html_to_markdown

from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.schemas.ingestion import (
    FetchMethod,
    FetchStats,
    RawDocument,
    UrlTarget,
)
from agent_knowledge_harvester.utils.token_counter import estimate_cost_usd, estimate_tokens

logger = logging.getLogger(__name__)


class Crawl4AIFallback:
    """Local extraction fallback for pages that Jina Reader cannot process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def fetch(self, target: UrlTarget) -> RawDocument:
        started = time.perf_counter()
        markdown = await self._fetch_with_crawl4ai(str(target.url))
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        markdown = markdown.strip()
        tokens = estimate_tokens(markdown)
        stats = FetchStats(
            method=FetchMethod.CRAWL4AI,
            source_url=target.url,
            elapsed_ms=elapsed_ms,
            input_chars=len(str(target.url)),
            output_chars=len(markdown),
            estimated_tokens=tokens,
            estimated_cost_usd=estimate_cost_usd(tokens),
        )

        if len(markdown) < self.settings.min_markdown_chars:
            stats.error = "Crawl4AI returned too little content"
            raise RuntimeError(stats.error)

        logger.info(
            "Fetched via Crawl4AI url=%s chars=%s tokens~=%s elapsed_ms=%s",
            target.url,
            len(markdown),
            tokens,
            elapsed_ms,
        )
        return RawDocument(
            source_url=target.url,
            method=FetchMethod.CRAWL4AI,
            markdown=markdown,
            stats=stats,
        )

    async def _fetch_with_crawl4ai(self, url: str) -> str:
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        except ImportError as exc:  # pragma: no cover - dependency is installed in full env
            raise RuntimeError("crawl4ai is not installed; run `uv sync` first") from exc

        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            user_agent=self.settings.user_agent,
        )
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            word_count_threshold=10,
            remove_overlay_elements=True,
            process_iframes=False,
            page_timeout=max(1000, int(self.settings.ingestion_timeout_seconds * 1000)),
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

        if not getattr(result, "success", False):
            error_message = getattr(result, "error_message", None) or "unknown Crawl4AI error"
            raise RuntimeError(error_message)

        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, str) and markdown.strip():
            return markdown

        html = getattr(result, "html", None)
        if isinstance(html, str) and html.strip():
            return html_to_markdown(html)

        raise RuntimeError("Crawl4AI produced no markdown or html")
