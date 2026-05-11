import asyncio
import logging
from pathlib import Path

from pydantic import HttpUrl, TypeAdapter

from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.ingestion.crawl4ai_fallback import Crawl4AIFallback
from agent_knowledge_harvester.ingestion.github_trending import GitHubTrendingClient
from agent_knowledge_harvester.ingestion.jina_reader import JinaReaderClient
from agent_knowledge_harvester.ingestion.preprocess import MarkdownPreprocessor
from agent_knowledge_harvester.schemas.ingestion import (
    CrawlRunStats,
    IngestionResult,
    RawDocument,
    SourceKind,
    UrlTarget,
)
from agent_knowledge_harvester.utils.files import stable_slug, write_json

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings, concurrency: int = 3) -> None:
        self.settings = settings
        self.jina_reader = JinaReaderClient(settings)
        self.crawl4ai = Crawl4AIFallback(settings)
        self.preprocessor = MarkdownPreprocessor(settings)
        self.trending = GitHubTrendingClient(settings)
        self.semaphore = asyncio.Semaphore(concurrency)

    async def ingest_urls(
        self,
        urls: list[str],
        out_dir: Path | None = None,
    ) -> list[IngestionResult]:
        targets = [
            UrlTarget(url=TypeAdapter(HttpUrl).validate_python(url), source_kind=SourceKind.URL)
            for url in urls
        ]
        return await self.ingest_targets(targets, out_dir=out_dir)

    async def ingest_github_trending(
        self,
        language: str | None = None,
        since: str = "daily",
        limit: int = 10,
        out_dir: Path | None = None,
    ) -> list[IngestionResult]:
        targets = await self.trending.discover(language=language, since=since, limit=limit)
        return await self.ingest_targets(targets, out_dir=out_dir)

    async def ingest_targets(
        self,
        targets: list[UrlTarget],
        out_dir: Path | None = None,
    ) -> list[IngestionResult]:
        tasks = [self._ingest_one(target, out_dir) for target in targets]
        results = await asyncio.gather(*tasks)
        run_stats = CrawlRunStats().finish(results)
        logger.info(
            "Ingestion finished total=%s succeeded=%s failed=%s tokens~=%s cost=$%.8f",
            run_stats.total_targets,
            run_stats.succeeded,
            run_stats.failed,
            run_stats.estimated_tokens,
            run_stats.estimated_cost_usd,
        )
        if out_dir:
            write_json(out_dir / "run_stats.json", run_stats.model_dump(mode="json"))
        return results

    async def _ingest_one(self, target: UrlTarget, out_dir: Path | None) -> IngestionResult:
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    self._fetch_and_clean(target),
                    timeout=self.settings.ingestion_timeout_seconds,
                )
                if out_dir:
                    self._persist_result(out_dir, result)
                return result
            except TimeoutError:
                error = (
                    "ingestion timed out after "
                    f"{self.settings.ingestion_timeout_seconds:.1f}s"
                )
                logger.warning("Ingestion timed out url=%s timeout=%s", target.url, error)
                result = IngestionResult(target=target, success=False, error=error)
                if out_dir:
                    self._persist_result(out_dir, result)
                return result
            except Exception as exc:
                logger.exception("Ingestion failed url=%s", target.url)
                result = IngestionResult(target=target, success=False, error=str(exc))
                if out_dir:
                    self._persist_result(out_dir, result)
                return result

    async def _fetch_and_clean(self, target: UrlTarget) -> IngestionResult:
        """Fetch one target and normalize it into a durable clean document."""
        raw: RawDocument | None = None
        try:
            try:
                raw = await self.jina_reader.fetch(target)
            except Exception as jina_exc:
                logger.warning("Jina Reader failed url=%s error=%r", target.url, jina_exc)
                raw = await self.crawl4ai.fetch(target)

            clean = self.preprocessor.clean(raw)
            return IngestionResult(target=target, raw=raw, clean=clean, success=True)
        except Exception as exc:
            if raw is not None:
                return IngestionResult(target=target, raw=raw, success=False, error=str(exc))
            raise

    def _persist_result(self, out_dir: Path, result: IngestionResult) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = stable_slug(str(result.target.url))

        if result.clean:
            md_path = out_dir / f"{slug}.md"
            md_path.write_text(result.clean.markdown, encoding="utf-8")

        write_json(out_dir / f"{slug}.json", result.model_dump(mode="json"))
