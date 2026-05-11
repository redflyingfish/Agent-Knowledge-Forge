from dataclasses import dataclass
from datetime import date

DEFAULT_DISCOVERY_CUTOFF = date(2026, 1, 1)
AUTHORITY_EXCEPTION_CUTOFF = date(2025, 6, 1)
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
    """Evaluate whether a source fits the project's current discovery window."""
    normalized_type = source_type.strip().lower()
    if published_at is None:
        if normalized_type in AUTHORITY_SOURCE_TYPES:
            return DiscoveryScopeDecision(
                in_scope=True,
                status="date_check_required_authority_source",
                reason=(
                    "Authority source without a known date; keep only if the page is current "
                    "or updated after the authority cutoff."
                ),
            )
        return DiscoveryScopeDecision(
            in_scope=False,
            status="date_check_required",
            reason="Non-authority source needs a date before it can enter expanded search.",
        )

    if published_at >= DEFAULT_DISCOVERY_CUTOFF:
        return DiscoveryScopeDecision(
            in_scope=True,
            status="in_scope_2026",
            reason="Default expanded search window accepts 2026 or newer sources.",
        )

    if published_at >= AUTHORITY_EXCEPTION_CUTOFF and normalized_type in AUTHORITY_SOURCE_TYPES:
        return DiscoveryScopeDecision(
            in_scope=True,
            status="in_scope_authority_exception",
            reason="Authority source is allowed from 2025-06-01 onward.",
        )

    if published_at >= AUTHORITY_EXCEPTION_CUTOFF and very_hot:
        return DiscoveryScopeDecision(
            in_scope=True,
            status="in_scope_very_hot_exception",
            reason="Very popular non-authority source is allowed from 2025-06-01 onward.",
        )

    return DiscoveryScopeDecision(
        in_scope=False,
        status="out_of_future_search_scope",
        reason=(
            "Future expanded search should skip this source unless the user explicitly "
            "keeps it as a legacy evaluation item."
        ),
    )
