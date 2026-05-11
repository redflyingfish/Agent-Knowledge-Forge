import logging

import httpx
from bs4 import BeautifulSoup

from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.schemas.ingestion import SourceKind, UrlTarget

logger = logging.getLogger(__name__)


class GitHubTrendingClient:
    """Discover repository URLs from GitHub Trending."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def discover(
        self,
        language: str | None = None,
        since: str = "daily",
        limit: int = 10,
    ) -> list[UrlTarget]:
        if since not in {"daily", "weekly", "monthly"}:
            raise ValueError("since must be one of: daily, weekly, monthly")

        path = f"/trending/{language.strip()}" if language else "/trending"
        url = f"https://github.com{path}?since={since}"
        headers = {"User-Agent": self.settings.user_agent}

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        targets: list[UrlTarget] = []
        for article in soup.select("article.Box-row"):
            anchor = article.select_one("h2 a")
            if not anchor or not anchor.get("href"):
                continue

            repo_url = f"https://github.com{anchor['href'].strip()}"
            targets.append(
                UrlTarget(
                    url=repo_url,
                    source_kind=SourceKind.GITHUB_TRENDING,
                    tags=[tag for tag in ["github", "trending", language, since] if tag],
                    discovered_from=url,
                )
            )
            if len(targets) >= limit:
                break

        logger.info("Discovered %s GitHub Trending repositories from %s", len(targets), url)
        return targets
