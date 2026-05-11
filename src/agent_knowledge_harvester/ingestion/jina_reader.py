import logging
import time
from urllib.parse import quote

import httpx

from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.schemas.ingestion import (
    FetchMethod,
    FetchStats,
    RawDocument,
    UrlTarget,
)
from agent_knowledge_harvester.utils.token_counter import estimate_cost_usd, estimate_tokens

logger = logging.getLogger(__name__)


class JinaReaderClient:
    """Fetch Markdown through Jina Reader using the cache-friendly reader endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def fetch(self, target: UrlTarget) -> RawDocument:
        source_url = str(target.url)
        reader_url = f"https://r.jina.ai/{quote(source_url, safe=':/?&=%#@+-._~')}"
        started = time.perf_counter()

        headers = {
            "Accept": "text/markdown",
            "User-Agent": self.settings.user_agent,
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.get(reader_url, headers=headers, follow_redirects=True)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        markdown = response.text.strip()
        tokens = estimate_tokens(markdown)
        stats = FetchStats(
            method=FetchMethod.JINA_READER,
            source_url=target.url,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            input_chars=len(source_url),
            output_chars=len(markdown),
            estimated_tokens=tokens,
            estimated_cost_usd=estimate_cost_usd(tokens),
        )

        if response.status_code >= 400:
            stats.error = f"Jina Reader returned HTTP {response.status_code}"
            raise RuntimeError(stats.error)

        if len(markdown) < self.settings.min_markdown_chars:
            stats.error = "Jina Reader returned too little content"
            raise RuntimeError(stats.error)

        logger.info(
            "Fetched via Jina Reader url=%s chars=%s tokens~=%s elapsed_ms=%s",
            target.url,
            len(markdown),
            tokens,
            elapsed_ms,
        )
        return RawDocument(
            source_url=target.url,
            final_url=str(response.url),
            method=FetchMethod.JINA_READER,
            markdown=markdown,
            stats=stats,
            metadata={"reader_url": reader_url},
        )
