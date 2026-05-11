import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Estimate token count with tiktoken, falling back to a conservative char ratio."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as exc:  # pragma: no cover - fallback depends on optional runtime data
        logger.debug("tiktoken unavailable, using char/4 fallback: %s", exc)
        return max(1, len(text) // 4)


def estimate_cost_usd(tokens: int, price_per_million_tokens: float = 0.0) -> float:
    """Keep cost accounting centralized, even for zero-cost preprocessing calls."""
    return round((tokens / 1_000_000) * price_per_million_tokens, 8)
