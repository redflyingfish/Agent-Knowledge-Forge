from dataclasses import dataclass
from datetime import date

DEFAULT_DISCOVERY_CUTOFF = date(2025, 1, 1)
AUTHORITY_SOURCE_TYPES = {
    "official_docs",
    "specification",
    "guide",
    "product_docs",
}


@dataclass(frozen=True)
class DiscoveryScopeDecision:
    in_scope: bool
    status: str
    reason: str


def evaluate_discovery_scope(
    source_type: str,
    published_at: date | None = None,
    very_hot: bool = False,
) -> DiscoveryScopeDecision:
    """Evaluate discovery recency as an advisory signal, not a hard gate."""
    normalized_type = source_type.strip().lower()
    if published_at is None:
        if normalized_type in AUTHORITY_SOURCE_TYPES:
            return DiscoveryScopeDecision(
                in_scope=True,
                status="date_unknown_authority_source",
                reason=(
                    "Authority source without a known date; keep it and let relevance, "
                    "authority, and deep reading decide its value."
                ),
            )
        return DiscoveryScopeDecision(
            in_scope=True,
            status="date_unknown_allowed",
            reason=(
                "Missing publication dates are common on GitHub, docs, and blogs; "
                "use date as ranking context rather than a rejection reason."
            ),
        )

    if published_at >= DEFAULT_DISCOVERY_CUTOFF:
        return DiscoveryScopeDecision(
            in_scope=True,
            status="in_scope_2025_plus",
            reason="Default expanded search window accepts 2025 or newer sources.",
        )

    return DiscoveryScopeDecision(
        in_scope=True,
        status="legacy_date_allowed",
        reason=(
            "Older sources are allowed when otherwise relevant; freshness should lower "
            "ranking but not block ingestion by itself."
        ),
    )
