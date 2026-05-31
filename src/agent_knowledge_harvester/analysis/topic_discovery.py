"""Mine candidate emerging topics from frontier discovery sources.

The search layer can find frontier, source-hub, and stop-signal sources, but it
does not know whether a source introduces a genuinely new concept.  This module
adds the missing middle step: extract candidate topics from fetched articles,
compare them with the fixed taxonomy, and report bounded coverage/yield metrics.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from agent_knowledge_harvester.analysis.knowledge_cards import (
    TOPIC_KEYWORDS,
    clamp_text,
    detect_topics,
    iter_ingestion_json_files,
    load_ingestion_result,
)
from agent_knowledge_harvester.discovery.search import normalize_result_url
from agent_knowledge_harvester.schemas.analysis import KnowledgeTopic
from agent_knowledge_harvester.schemas.discovery import SearchDiscoveryReport
from agent_knowledge_harvester.schemas.ingestion import CleanDocument
from agent_knowledge_harvester.utils.files import write_json

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "best",
    "by",
    "for",
    "from",
    "guide",
    "how",
    "in",
    "into",
    "is",
    "it",
    "latest",
    "new",
    "of",
    "on",
    "or",
    "overview",
    "the",
    "to",
    "use",
    "using",
    "vs",
    "with",
    "what",
    "why",
    "do",
    "does",
    "can",
    "you",
}

QUESTION_PREFIXES = ("why ", "what ", "how ", "when ", "where ", "who ")
LOW_VALUE_TOPIC_PHRASES = {
    "competitive implications",
    "conclusion",
    "implementation details",
    "introduction",
    "key takeaways",
    "overview",
    "summary",
    "technical architecture and implementation",
}
HEADLINE_VERBS = ("announce", "announces", "introduce", "introduces", "launch", "launches")
LOW_VALUE_TOPIC_SUBSTRINGS = (
    " calls ",
    " executes ",
    " passed ",
    " returns ",
    " was ",
)
KNOWN_SHORT_TOPIC_ALIASES = {
    "agent loop": KnowledgeTopic.AGENT_ARCHITECTURE.value,
    "agent skills": KnowledgeTopic.SKILLS.value,
    "claude code": KnowledgeTopic.CLAUDE_CODE.value,
    "data source": KnowledgeTopic.DATA_CONNECTORS.value,
    "data sources": KnowledgeTopic.DATA_CONNECTORS.value,
    "openclaw": KnowledgeTopic.OPENCLAW.value,
    "skill file": KnowledgeTopic.SKILLS.value,
    "skills": KnowledgeTopic.SKILLS.value,
    "technical architecture": KnowledgeTopic.AGENT_ARCHITECTURE.value,
}

TOPIC_HINT_PATTERNS = (
    re.compile(r"\b(?:emerging|frontier|novel|recent)\s+([a-z][a-z0-9 /_-]{3,80})", re.I),
    re.compile(
        r"\b(?:benchmark|leaderboard|survey|retrospective)\s+(?:for|of|on)\s+"
        r"([a-z][a-z0-9 /_-]{3,80})",
        re.I,
    ),
    re.compile(
        r"\b(?:challenge|risk|failure mode|gap|limitation)s?\s+(?:in|for|of)\s+"
        r"([a-z][a-z0-9 /_-]{3,80})",
        re.I,
    ),
    re.compile(
        r"\b(?:evaluation|eval|observability|routing|memory|context|tool use|mcp|guardrail)s?"
        r"\s+(?:for|in)\s+([a-z][a-z0-9 /_-]{3,80})",
        re.I,
    ),
)
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,4}\s+(.+?)\s*$")
LIST_TOPIC_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")
WORD_PATTERN = re.compile(r"[a-z][a-z0-9_-]+", re.I)


@dataclass(frozen=True)
class TopicObservation:
    topic: str
    normalized_topic: str
    source_url: str = ""
    source_title: str = ""
    evidence: str = ""
    confidence: float = 0.5
    source_bucket: str = "unknown"


@dataclass(frozen=True)
class TopicCandidate:
    topic: str
    normalized_topic: str
    source_count: int
    mention_count: int
    confidence: float
    source_buckets: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    nearest_known_topic: str | None = None
    nearest_known_similarity: float = 0.0
    is_new: bool = True
    promoted: bool = False
    suggested_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicDiscoveryMetrics:
    known_topic_count: int
    observed_topic_count: int
    new_topic_count: int
    promoted_topic_count: int
    covered_known_topic_count: int
    known_topic_coverage: float
    new_topic_rate: float
    low_yield: bool


@dataclass(frozen=True)
class TopicDiscoveryReport:
    candidates: tuple[TopicCandidate, ...]
    metrics: TopicDiscoveryMetrics
    covered_known_topics: tuple[str, ...] = ()
    stop_recommendation: bool = False
    stop_reason: str = ""
    warnings: tuple[str, ...] = ()


async def discover_topics_from_ingestion_dir(
    in_dir: Path,
    *,
    llm_client: object | None = None,
    use_llm: bool = False,
    search_report_path: Path | None = None,
    max_docs: int | None = None,
    max_topics_per_doc: int = 8,
    min_sources: int = 2,
    min_confidence: float = 0.55,
    known_similarity_threshold: float = 0.78,
    concurrency: int = 2,
) -> TopicDiscoveryReport:
    """Extract and aggregate candidate topics from ingestion JSON artifacts."""

    source_buckets = load_source_buckets(search_report_path) if search_report_path else {}
    paths = list(iter_ingestion_json_files(in_dir))
    if max_docs is not None:
        paths = paths[:max_docs]

    semaphore = asyncio.Semaphore(max(1, concurrency))
    observations: list[TopicObservation] = []
    covered_known_topics: set[KnowledgeTopic] = set()
    warnings: list[str] = []

    async def analyze_path(
        path: Path,
    ) -> tuple[list[TopicObservation], set[KnowledgeTopic], str | None]:
        result = load_ingestion_result(path)
        if result is None or not result.success or result.clean is None:
            return [], set(), None
        document = result.clean
        source_url = str(document.source_url)
        source_bucket = source_buckets.get(normalize_result_url(source_url), "unknown")
        known_hits = set(detect_topics(document.markdown))
        async with semaphore:
            try:
                doc_observations = await extract_document_topics(
                    document,
                    llm_client=llm_client,
                    use_llm=use_llm,
                    source_bucket=source_bucket,
                    max_topics=max_topics_per_doc,
                )
            except Exception as exc:  # noqa: BLE001 - topic mining should degrade to metrics.
                return [], known_hits, f"{source_url}: {type(exc).__name__}: {str(exc)[:180]}"
        return list(doc_observations), known_hits, None

    for doc_observations, known_hits, warning in await asyncio.gather(
        *(analyze_path(path) for path in paths)
    ):
        observations.extend(doc_observations)
        covered_known_topics.update(known_hits)
        if warning:
            warnings.append(warning)

    report = discover_topics(
        observations,
        known_topics=[topic.value for topic in KnowledgeTopic],
        covered_known_topics=[topic.value for topic in covered_known_topics],
        min_sources=min_sources,
        min_confidence=min_confidence,
        known_similarity_threshold=known_similarity_threshold,
    )
    return TopicDiscoveryReport(
        candidates=report.candidates,
        metrics=report.metrics,
        covered_known_topics=report.covered_known_topics,
        stop_recommendation=report.stop_recommendation,
        stop_reason=report.stop_reason,
        warnings=tuple(warnings),
    )


async def extract_document_topics(
    document: CleanDocument,
    *,
    llm_client: object | None = None,
    use_llm: bool = False,
    source_bucket: str = "unknown",
    max_topics: int = 8,
) -> tuple[TopicObservation, ...]:
    """Extract topic observations from one clean document."""

    source_url = str(document.source_url)
    title = document.title or ""
    heuristic = extract_candidate_topics_from_text(
        document.markdown,
        source_url=source_url,
        source_title=title,
        source_bucket=source_bucket,
        max_topics=max_topics,
    )
    if not use_llm:
        return heuristic
    if llm_client is None:
        raise RuntimeError("use_llm=True requires an llm_client")

    prompt = build_topic_discovery_user_prompt(
        document,
        max_topics=max_topics,
    )
    llm_result = await llm_client.chat_json(
        "extraction",
        system_prompt=build_topic_discovery_system_prompt(),
        user_prompt=prompt,
        temperature=0.0,
    )
    llm_observations = parse_llm_topic_payload(
        llm_result.payload,
        source_url=source_url,
        source_title=title,
        source_bucket=source_bucket,
        max_topics=max_topics,
    )
    return llm_observations or heuristic


def build_topic_discovery_system_prompt() -> str:
    return (
        "You extract source-grounded emerging topics for AI-agent engineering. "
        "Return strict JSON only. Do not include markdown fences or commentary."
    )


def build_topic_discovery_user_prompt(document: CleanDocument, *, max_topics: int = 8) -> str:
    known = ", ".join(topic.value for topic in KnowledgeTopic)
    markdown = clamp_text(document.markdown, 12_000)
    return (
        "Extract candidate new or under-covered AI-agent engineering topics from this article.\n"
        "A good candidate is more specific than the fixed taxonomy and is useful for follow-up "
        "search, interviews, or implementation decisions.\n"
        f"Allowed fixed taxonomy for comparison: {known}\n"
        f"Return at most {max_topics} topics.\n"
        "JSON schema:\n"
        "{\n"
        '  "topics": [\n'
        "    {\n"
        '      "topic": "short noun phrase under 8 words",\n'
        '      "confidence": 0.0,\n'
        '      "evidence": "short supporting phrase from the article"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Source URL: {document.source_url}\n"
        f"Title: {document.title or '(none)'}\n"
        f"Summary hint: {document.summary_hint}\n\n"
        f"Article markdown:\n{markdown}"
    )


def parse_llm_topic_payload(
    payload: Mapping[str, Any],
    *,
    source_url: str = "",
    source_title: str = "",
    source_bucket: str = "llm",
    max_topics: int = 8,
) -> tuple[TopicObservation, ...]:
    raw_topics = payload.get("topics", [])
    if not isinstance(raw_topics, list):
        return ()

    observations: list[TopicObservation] = []
    for item in raw_topics[:max_topics]:
        if not isinstance(item, Mapping):
            continue
        topic = str(item.get("topic", "")).strip()
        normalized = normalize_topic(topic)
        if not is_valid_topic(normalized):
            continue
        confidence = coerce_float(item.get("confidence", 0.65), default=0.65)
        observations.append(
            TopicObservation(
                topic=topic,
                normalized_topic=normalized,
                source_url=source_url,
                source_title=source_title,
                evidence=clamp_text(str(item.get("evidence") or ""), 280),
                confidence=max(0.0, min(1.0, confidence)),
                source_bucket=source_bucket,
            )
        )
    return tuple(observations)


def extract_candidate_topics_from_text(
    text: str,
    *,
    source_url: str = "",
    source_title: str = "",
    source_bucket: str = "heuristic",
    max_topics: int = 8,
) -> tuple[TopicObservation, ...]:
    """Cheap deterministic topic extraction used as fallback and smoke test."""

    observations: list[TopicObservation] = []
    seen: set[str] = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        heading_match = HEADING_PATTERN.match(line)
        list_match = LIST_TOPIC_PATTERN.match(line)
        phrase = ""
        confidence = 0.0
        if heading_match:
            phrase = trim_candidate_phrase(heading_match.group(1))
            if not looks_like_topic_line(phrase) or is_low_value_topic_phrase(phrase):
                continue
            confidence = 0.55
        elif list_match:
            phrase = trim_candidate_phrase(list_match.group(1))
            if not looks_like_topic_line(phrase) or is_low_value_topic_phrase(phrase):
                continue
            confidence = 0.45
        if phrase:
            append_observation(
                observations,
                seen,
                phrase,
                source_url=source_url,
                source_title=source_title,
                source_bucket=source_bucket,
                evidence=line,
                confidence=confidence,
                max_topics=max_topics,
            )
        if len(observations) >= max_topics:
            break

    for pattern in TOPIC_HINT_PATTERNS:
        if len(observations) >= max_topics:
            break
        for match in pattern.finditer(text):
            phrase = trim_candidate_phrase(match.group(1))
            if not looks_like_topic_line(phrase) or is_low_value_topic_phrase(phrase):
                continue
            append_observation(
                observations,
                seen,
                phrase,
                source_url=source_url,
                source_title=source_title,
                source_bucket=source_bucket,
                evidence=match.group(0),
                confidence=0.6,
                max_topics=max_topics,
            )
            if len(observations) >= max_topics:
                break

    return tuple(observations)


def discover_topics(
    observations: Iterable[TopicObservation],
    *,
    known_topics: Sequence[str],
    covered_known_topics: Sequence[str] = (),
    min_sources: int = 2,
    min_confidence: float = 0.55,
    known_similarity_threshold: float = 0.78,
    max_candidates: int = 40,
    low_yield_new_topics: int = 1,
) -> TopicDiscoveryReport:
    """Aggregate observations and compute conservative coverage/yield metrics."""

    grouped: dict[str, list[TopicObservation]] = defaultdict(list)
    for observation in observations:
        if is_valid_topic(observation.normalized_topic):
            grouped[observation.normalized_topic].append(observation)

    known_norms = {normalize_topic(topic.replace("_", " ")): topic for topic in known_topics}
    candidates: list[TopicCandidate] = []
    matched_known_topics: set[str] = set(covered_known_topics)

    for normalized, rows in grouped.items():
        topic_label = best_label(rows)
        nearest_known_topic, nearest_similarity = nearest_known_topic_match(normalized, known_norms)
        direct_known_topic = direct_known_topic_match(topic_label)
        if direct_known_topic and len(normalized.split()) <= 3:
            nearest_known_topic = direct_known_topic
            nearest_similarity = max(nearest_similarity, 0.82)
        is_new = nearest_similarity < known_similarity_threshold
        source_urls = tuple(unique(row.source_url for row in rows if row.source_url))
        source_buckets = tuple(unique(row.source_bucket for row in rows if row.source_bucket))
        source_count = (
            len(source_urls)
            if source_urls
            else len(unique(row.source_title for row in rows))
        )
        source_count = source_count or 1
        confidence = sum(row.confidence for row in rows) / len(rows)
        promoted = is_new and source_count >= min_sources and confidence >= min_confidence
        if not is_new and nearest_known_topic:
            matched_known_topics.add(nearest_known_topic)
        candidates.append(
            TopicCandidate(
                topic=topic_label,
                normalized_topic=normalized,
                source_count=source_count,
                mention_count=len(rows),
                confidence=round(confidence, 3),
                source_buckets=source_buckets,
                source_urls=source_urls[:5],
                evidence=tuple(unique(row.evidence for row in rows if row.evidence))[:3],
                nearest_known_topic=nearest_known_topic,
                nearest_known_similarity=round(nearest_similarity, 3),
                is_new=is_new,
                promoted=promoted,
                suggested_queries=tuple(suggest_queries(topic_label)) if promoted else (),
            )
        )

    candidates.sort(
        key=lambda item: (
            item.promoted,
            item.is_new,
            item.source_count,
            item.mention_count,
            item.confidence,
        ),
        reverse=True,
    )
    candidates = candidates[:max_candidates]
    observed_count = len(candidates)
    new_count = sum(1 for item in candidates if item.is_new)
    promoted_count = sum(1 for item in candidates if item.promoted)
    coverage = len(matched_known_topics) / max(1, len(set(known_topics)))
    new_rate = new_count / max(1, observed_count)
    metrics = TopicDiscoveryMetrics(
        known_topic_count=len(set(known_topics)),
        observed_topic_count=observed_count,
        new_topic_count=new_count,
        promoted_topic_count=promoted_count,
        covered_known_topic_count=len(matched_known_topics),
        known_topic_coverage=round(coverage, 3),
        new_topic_rate=round(new_rate, 3),
        low_yield=promoted_count <= low_yield_new_topics,
    )
    return TopicDiscoveryReport(
        candidates=tuple(candidates),
        metrics=metrics,
        covered_known_topics=tuple(sorted(matched_known_topics)),
    )


def recommend_stop_from_batches(
    reports: Sequence[TopicDiscoveryReport],
    *,
    low_yield_batches: int = 3,
    min_known_topic_coverage: float = 0.65,
) -> TopicDiscoveryReport:
    """Return the latest report with a conservative stop recommendation attached."""

    if not reports:
        raise ValueError("reports must not be empty")
    latest = reports[-1]
    recent = reports[-low_yield_batches:]
    should_stop = (
        len(recent) >= low_yield_batches
        and all(report.metrics.low_yield for report in recent)
        and latest.metrics.known_topic_coverage >= min_known_topic_coverage
    )
    reason = ""
    if should_stop:
        reason = (
            f"{low_yield_batches} consecutive batches had low promoted-topic yield "
            f"and known-topic coverage reached {latest.metrics.known_topic_coverage:.0%}."
        )
    return TopicDiscoveryReport(
        candidates=latest.candidates,
        metrics=latest.metrics,
        covered_known_topics=latest.covered_known_topics,
        stop_recommendation=should_stop,
        stop_reason=reason,
        warnings=latest.warnings,
    )


def load_source_buckets(search_report_path: Path) -> dict[str, str]:
    """Map normalized result URLs to the query family that discovered them."""

    report = SearchDiscoveryReport.model_validate_json(
        search_report_path.read_text(encoding="utf-8")
    )
    query_buckets: dict[str, str] = {}
    for query in report.query_plan.frontier_scout_queries:
        query_buckets[query] = "frontier_scout"
    for query in report.query_plan.stop_signal_queries:
        query_buckets[query] = "stop_signal"
    for query in report.query_plan.source_hub_queries:
        query_buckets[query] = "source_hub"
    for expansion in report.query_plan.topic_expansions:
        for query in expansion.authority_queries:
            query_buckets[query] = "topic_authority"
        for query in expansion.implementation_queries:
            query_buckets[query] = "topic_implementation"
        for query in expansion.risk_queries:
            query_buckets[query] = "topic_risk"

    source_buckets: dict[str, str] = {}
    for result in report.results:
        normalized_url = normalize_result_url(result.url)
        if normalized_url and normalized_url not in source_buckets:
            source_buckets[normalized_url] = query_buckets.get(result.query, "unknown")
    return source_buckets


def write_topic_discovery_report(report: TopicDiscoveryReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "topic_discovery_report.json", report_to_dict(report))
    (out_dir / "topic_discovery_report.md").write_text(
        render_topic_discovery_report(report),
        encoding="utf-8",
    )


def render_topic_discovery_report(report: TopicDiscoveryReport) -> str:
    lines = [
        "# Topic Discovery Report",
        "",
        "## Metrics",
        "",
        f"- Known topic coverage: {report.metrics.known_topic_coverage:.1%} "
        f"({report.metrics.covered_known_topic_count}/{report.metrics.known_topic_count})",
        f"- Observed candidate topics: {report.metrics.observed_topic_count}",
        f"- New candidate topics: {report.metrics.new_topic_count}",
        f"- Promoted candidate topics: {report.metrics.promoted_topic_count}",
        f"- New topic rate: {report.metrics.new_topic_rate:.1%}",
        f"- Low-yield batch: {report.metrics.low_yield}",
        f"- Stop recommendation: {report.stop_recommendation}",
    ]
    if report.stop_reason:
        lines.append(f"- Stop reason: {report.stop_reason}")
    if report.covered_known_topics:
        lines.extend(["", "## Covered Known Topics", ""])
        lines.append(", ".join(report.covered_known_topics))
    summary = bucket_summary(report)
    if summary:
        lines.extend(["", "## Source Bucket Contribution", ""])
        lines.extend(
            f"- {bucket}: candidates={values['candidate_count']}, "
            f"promoted={values['promoted_count']}"
            for bucket, values in summary.items()
        )
    lines.extend(["", "## Candidate Topics", ""])
    for candidate in report.candidates:
        status = "promoted" if candidate.promoted else "candidate"
        if not candidate.is_new:
            status = f"known-ish: {candidate.nearest_known_topic}"
        lines.extend(
            [
                f"### {candidate.topic}",
                "",
                f"- Status: {status}",
                f"- Confidence: {candidate.confidence}",
                f"- Sources: {candidate.source_count}; mentions: {candidate.mention_count}",
                f"- Source buckets: {', '.join(candidate.source_buckets) or '(unknown)'}",
                f"- Nearest known topic: {candidate.nearest_known_topic or '(none)'} "
                f"({candidate.nearest_known_similarity})",
            ]
        )
        if candidate.suggested_queries:
            lines.append("- Suggested queries:")
            lines.extend(f"  - `{query}`" for query in candidate.suggested_queries)
        if candidate.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {evidence}" for evidence in candidate.evidence)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def report_to_dict(report: TopicDiscoveryReport) -> dict[str, Any]:
    return {
        "metrics": report.metrics.__dict__,
        "covered_known_topics": list(report.covered_known_topics),
        "source_bucket_summary": bucket_summary(report),
        "stop_recommendation": report.stop_recommendation,
        "stop_reason": report.stop_reason,
        "warnings": list(report.warnings),
        "candidates": [
            {
                "topic": item.topic,
                "normalized_topic": item.normalized_topic,
                "source_count": item.source_count,
                "mention_count": item.mention_count,
                "confidence": item.confidence,
                "source_buckets": list(item.source_buckets),
                "source_urls": list(item.source_urls),
                "evidence": list(item.evidence),
                "nearest_known_topic": item.nearest_known_topic,
                "nearest_known_similarity": item.nearest_known_similarity,
                "is_new": item.is_new,
                "promoted": item.promoted,
                "suggested_queries": list(item.suggested_queries),
            }
            for item in report.candidates
        ],
    }


def bucket_summary(report: TopicDiscoveryReport) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for candidate in report.candidates:
        buckets = candidate.source_buckets or ("unknown",)
        for bucket in buckets:
            values = summary.setdefault(bucket, {"candidate_count": 0, "promoted_count": 0})
            values["candidate_count"] += 1
            if candidate.promoted:
                values["promoted_count"] += 1
    return dict(sorted(summary.items()))


def normalize_topic(value: str) -> str:
    words = WORD_PATTERN.findall(value.lower().replace("&", " and "))
    words = [word.strip("_-") for word in words if word.strip("_-")]
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words[:8]).strip()


def topic_similarity(left: str, right: str) -> float:
    left_norm = normalize_topic(left)
    right_norm = normalize_topic(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_words = set(left_norm.split())
    right_words = set(right_norm.split())
    jaccard = len(left_words & right_words) / max(1, len(left_words | right_words))
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, sequence * 0.85)


def append_observation(
    observations: list[TopicObservation],
    seen: set[str],
    phrase: str,
    *,
    source_url: str,
    source_title: str,
    source_bucket: str,
    evidence: str,
    confidence: float,
    max_topics: int,
) -> None:
    normalized = normalize_topic(trim_candidate_phrase(phrase))
    trimmed = trim_candidate_phrase(phrase)
    if (
        len(observations) >= max_topics
        or normalized in seen
        or is_question_heading(trimmed)
        or not is_valid_topic(normalized)
    ):
        return
    normalized_words = set(normalized.split())
    if any(normalized_words < set(existing.split()) for existing in seen):
        return
    seen.add(normalized)
    observations.append(
        TopicObservation(
            topic=trim_candidate_phrase(phrase),
            normalized_topic=normalized,
            source_url=source_url,
            source_title=source_title,
            source_bucket=source_bucket,
            evidence=clamp_text(evidence, 280),
            confidence=confidence,
        )
    )


def best_label(rows: Sequence[TopicObservation]) -> str:
    counts = Counter(row.topic.strip() for row in rows if row.topic.strip())
    if not counts:
        return rows[0].normalized_topic
    return counts.most_common(1)[0][0]


def coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_valid_topic(normalized: str) -> bool:
    if not normalized:
        return False
    words = normalized.split()
    if len(words) < 2 or len(words) > 8:
        return False
    return sum(len(word) for word in words) >= 8


def is_question_heading(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.endswith("?") or lowered.startswith(QUESTION_PREFIXES)


def direct_known_topic_match(value: str) -> str | None:
    normalized = normalize_topic(value)
    if normalized in KNOWN_SHORT_TOPIC_ALIASES:
        return KNOWN_SHORT_TOPIC_ALIASES[normalized]
    lowered = value.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic is KnowledgeTopic.AGENT_ARCHITECTURE:
            continue
        if any(keyword in lowered for keyword in keywords):
            return topic.value
    return None


def is_low_value_topic_phrase(value: str) -> bool:
    lowered = value.strip().lower()
    normalized = normalize_topic(lowered)
    if normalized in LOW_VALUE_TOPIC_PHRASES or lowered in LOW_VALUE_TOPIC_PHRASES:
        return True
    if any(verb in lowered.split() for verb in HEADLINE_VERBS):
        return True
    return any(substring in f" {lowered} " for substring in LOW_VALUE_TOPIC_SUBSTRINGS)


def looks_like_topic_line(value: str) -> bool:
    normalized = normalize_topic(value)
    if not is_valid_topic(normalized):
        return False
    lower = value.lower()
    return any(
        marker in lower
        for marker in (
            "agent",
            "benchmark",
            "context",
            "eval",
            "guardrail",
            "mcp",
            "memory",
            "openclaw",
            "routing",
            "runtime",
            "skill",
            "tool",
            "workflow",
        )
    )


def nearest_known_topic_match(
    normalized: str,
    known_norms: Mapping[str, str],
) -> tuple[str | None, float]:
    best_topic: str | None = None
    best_score = 0.0
    for known_norm, known_topic in known_norms.items():
        score = topic_similarity(normalized, known_norm)
        if score > best_score:
            best_score = score
            best_topic = known_topic
    return best_topic, best_score


def suggest_queries(topic: str) -> list[str]:
    clean = " ".join(topic.split())
    return [
        f"{clean} AI agents 2026",
        f"{clean} agent systems benchmark 2026",
        f"{clean} official docs technical report",
    ]


def trim_candidate_phrase(value: str) -> str:
    value = value.replace("**", "").replace("__", "")
    value = re.split(r"\s+-\s+", value, maxsplit=1)[0]
    phrase = re.split(r"[.;:()\[\]\n]", value, maxsplit=1)[0]
    phrase = re.sub(r"\s+", " ", phrase).strip(" -_*`'\"")
    words = phrase.split()
    if len(words) > 8:
        phrase = " ".join(words[:8])
    return phrase


def unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)
