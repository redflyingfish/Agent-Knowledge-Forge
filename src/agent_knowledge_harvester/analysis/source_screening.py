import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from agent_knowledge_harvester.analysis.discovery_policy import evaluate_discovery_scope
from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex
from agent_knowledge_harvester.schemas.ingestion import UrlTarget
from agent_knowledge_harvester.schemas.screening import (
    LLMScreeningJudgment,
    ScreenedSource,
    ScreeningDecision,
    ScreeningReport,
    SourceCandidate,
)
from agent_knowledge_harvester.utils.files import write_json

CORE_AGENT_TERMS = {
    "agent",
    "agents",
    "agent-native",
    "ai agent",
    "ai coding agent",
    "ai coding agents",
    "mcp",
    "model context protocol",
    "langgraph",
    "multi-agent",
}

SUPPORTING_AGENT_TERMS = {
    "tool calling",
    "function calling",
    "workflow",
    "orchestration",
    "rag",
    "memory",
    "autonomous",
    "assistant",
    "connector",
}

NOVELTY_COMMON_TERMS = {
    "agent",
    "agents",
    "agent-native",
    "ai",
    "mcp",
    "model",
    "context",
    "protocol",
    "langgraph",
    "multi-agent",
    "tool",
    "tools",
    "workflow",
    "workflows",
    "memory",
    "retrieval",
    "rag",
    "server",
    "servers",
    "client",
    "clients",
    "connector",
    "connectors",
    "application",
    "applications",
    "system",
    "systems",
    "framework",
    "frameworks",
    "build",
    "building",
    "using",
    "with",
    "for",
    "from",
    "that",
    "this",
}

AUTHORITY_DOMAIN_SCORES = {
    "modelcontextprotocol.io": 0.86,
    "docs.anthropic.com": 0.84,
    "anthropic.com": 0.78,
    "platform.openai.com": 0.84,
    "openai.com": 0.78,
    "langchain-ai.github.io": 0.78,
    "langchain.com": 0.76,
    "github.blog": 0.68,
}


class GitHubRepoMetadataClient:
    """Fetch GitHub repository metadata used for source screening."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def enrich(self, target: UrlTarget) -> SourceCandidate:
        owner, repo = parse_github_repo(str(target.url))
        fallback = SourceCandidate(
            url=target.url,
            title=f"{owner}/{repo}" if owner and repo else str(target.url),
            source_kind=target.source_kind.value,
            discovered_from=target.discovered_from,
            author=owner,
            topics=target.tags,
        )
        if not owner or not repo:
            return fallback

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.settings.user_agent,
        }
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(api_url, headers=headers)
                if response.status_code >= 400:
                    return fallback
                payload = response.json()
        except httpx.HTTPError:
            return fallback

        return SourceCandidate(
            url=target.url,
            title=payload.get("full_name") or fallback.title,
            summary=payload.get("description") or "",
            source_kind=target.source_kind.value,
            discovered_from=target.discovered_from,
            author=owner,
            repo_stars=payload.get("stargazers_count"),
            repo_forks=payload.get("forks_count"),
            owner_followers=None,
            pushed_at=parse_datetime(payload.get("pushed_at")),
            topics=payload.get("topics") or target.tags,
        )


class GenericUrlMetadataClient:
    """Fetch lightweight metadata for non-GitHub article, docs, and spec URLs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def enrich(self, target: UrlTarget) -> SourceCandidate:
        fallback = SourceCandidate(
            url=target.url,
            title=title_from_url(str(target.url)),
            source_kind=target.source_kind.value,
            discovered_from=target.discovered_from,
            author=domain_from_url(str(target.url)),
            topics=topics_from_url(str(target.url)) + target.tags,
        )
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(
                    str(target.url),
                    headers={"User-Agent": self.settings.user_agent},
                    follow_redirects=True,
                )
                if response.status_code >= 400:
                    return fallback
        except httpx.HTTPError:
            return fallback

        soup = BeautifulSoup(response.text, "html.parser")
        title = first_text(
            soup.title.string if soup.title and soup.title.string else None,
            meta_content(soup, "og:title"),
            meta_content(soup, "twitter:title"),
            fallback.title,
        )
        summary = first_text(
            meta_content(soup, "description"),
            meta_content(soup, "og:description"),
            meta_content(soup, "twitter:description"),
            "",
        )
        published_at = first_datetime(
            meta_content(soup, "article:published_time"),
            meta_content(soup, "article:modified_time"),
            meta_content(soup, "date"),
            meta_content(soup, "datePublished"),
            meta_content(soup, "publish_date"),
        )
        author = first_text(meta_content(soup, "author"), fallback.author)
        topics = unique_texts(topics_from_url(str(target.url)) + target.tags)

        return SourceCandidate(
            url=target.url,
            title=title,
            summary=summary,
            source_kind=target.source_kind.value,
            discovered_from=target.discovered_from,
            author=author,
            pushed_at=published_at,
            topics=topics,
        )


def screen_candidates(
    candidates: list[SourceCandidate],
    existing_index: KnowledgeIndex | None = None,
    min_accept_score: float = 0.5,
    min_relevance_score: float = 0.16,
) -> ScreeningReport:
    existing_texts = build_existing_memory_texts(existing_index)
    screened = [
        score_candidate(
            candidate,
            existing_texts=existing_texts,
            min_accept_score=min_accept_score,
            min_relevance_score=min_relevance_score,
        )
        for candidate in candidates
    ]
    screened.sort(key=lambda item: item.overall_score, reverse=True)
    selected_urls = [
        item.candidate.url for item in screened if item.decision == ScreeningDecision.ACCEPT
    ]
    return ScreeningReport(
        total_candidates=len(screened),
        accepted=sum(1 for item in screened if item.decision == ScreeningDecision.ACCEPT),
        review=sum(1 for item in screened if item.decision == ScreeningDecision.REVIEW),
        rejected=sum(1 for item in screened if item.decision == ScreeningDecision.REJECT),
        selected_urls=selected_urls,
        sources=screened,
    )


async def refine_screening_with_llm(
    report: ScreeningReport,
    llm_client: object,
    existing_index: KnowledgeIndex | None = None,
    stage: str = "screening",
    max_candidates: int = 10,
) -> ScreeningReport:
    """Use an LLM as a semantic gatekeeper after deterministic scoring."""
    existing_texts = build_existing_memory_texts(existing_index)
    selected = select_llm_review_sources(report.sources, max_candidates=max_candidates)
    for source in selected:
        result = await llm_client.chat_json(
            stage,
            system_prompt=build_llm_screening_system_prompt(),
            user_prompt=build_llm_screening_user_prompt(source, existing_texts),
            temperature=0.0,
        )
        judgment = parse_llm_screening_judgment(result.payload)
        judgment.model = result.model
        judgment.prompt_tokens_estimate = result.prompt_tokens_estimate
        judgment.completion_tokens_estimate = result.completion_tokens_estimate
        apply_llm_judgment(source, judgment)

    report.llm_enabled = True
    report.llm_judged = len(selected)
    recalculate_report_counts(report)
    report.sources.sort(key=lambda item: item.overall_score, reverse=True)
    return report


def select_llm_review_sources(
    sources: list[ScreenedSource],
    max_candidates: int,
) -> list[ScreenedSource]:
    """Prioritize likely accepts and borderline rejects for paid semantic review."""
    if max_candidates <= 0:
        return []
    priority = {
        ScreeningDecision.ACCEPT: 0,
        ScreeningDecision.REVIEW: 1,
        ScreeningDecision.REJECT: 2,
    }
    ranked = sorted(
        sources,
        key=lambda item: (
            priority[item.decision],
            abs(item.relevance_score - 0.16),
            -item.authority_score,
            -item.freshness_score,
        ),
    )
    return ranked[:max_candidates]


def build_llm_screening_system_prompt() -> str:
    return (
        "You are a strict source-screening judge for an agent-development knowledge harvester. "
        "Accept only sources that teach reusable, frontier-relevant agent engineering knowledge: "
        "agent architecture, tool use, MCP, memory/retrieval, evaluation, orchestration, or "
        "multi-agent systems. Reject application-only projects, generic automation, marketing, "
        "duplicative material, or weakly reliable sources. Return strict JSON only."
    )


def build_llm_screening_user_prompt(
    source: ScreenedSource,
    existing_memory_texts: list[str],
    max_memory_items: int = 8,
) -> str:
    candidate = source.candidate
    memory_items = existing_memory_texts[:max_memory_items]
    payload = {
        "candidate": {
            "url": str(candidate.url),
            "title": candidate.title,
            "summary": candidate.summary,
            "source_kind": candidate.source_kind,
            "author": candidate.author,
            "repo_stars": candidate.repo_stars,
            "repo_forks": candidate.repo_forks,
            "pushed_at": candidate.pushed_at.isoformat() if candidate.pushed_at else None,
            "topics": candidate.topics,
        },
        "deterministic_screen": {
            "decision": source.decision.value,
            "overall_score": source.overall_score,
            "relevance_score": source.relevance_score,
            "authority_score": source.authority_score,
            "freshness_score": source.freshness_score,
            "novelty_score": source.novelty_score,
            "reasons": source.reasons,
        },
        "existing_memory_fingerprints": memory_items,
        "task": (
            "Judge whether this source should enter a compact agent-readable memory. "
            "Use title, summary, topics, popularity, recency, and memory overlap. "
            "decision must be one of accept, review, reject. "
            "memory_action must be ingest, watch, or skip. "
            "reasoning must be one short sentence under 25 words."
        ),
        "response_schema": {
            "decision": "accept|review|reject",
            "agent_relevance": "0.0-1.0",
            "reliability": "0.0-1.0",
            "novelty": "0.0-1.0",
            "classic_value": "0.0-1.0",
            "memory_action": "ingest|watch|skip",
            "reasoning": "short reason",
            "concerns": ["short concern"],
        },
    }
    import json

    return json.dumps(payload, ensure_ascii=False)


def parse_llm_screening_judgment(payload: dict[str, object]) -> LLMScreeningJudgment:
    raw = payload.get("judgment", payload)
    if not isinstance(raw, dict):
        raise ValueError("LLM screening response must be a JSON object")
    normalized = {
        "decision": normalize_llm_decision(raw.get("decision")),
        "agent_relevance": clamp_float(raw.get("agent_relevance")),
        "reliability": clamp_float(raw.get("reliability")),
        "novelty": clamp_float(raw.get("novelty")),
        "classic_value": clamp_float(raw.get("classic_value")),
        "memory_action": normalize_memory_action(raw.get("memory_action")),
        "reasoning": str(raw.get("reasoning") or "")[:240],
        "concerns": normalize_concerns(raw.get("concerns")),
    }
    return LLMScreeningJudgment.model_validate(normalized)


def apply_llm_judgment(source: ScreenedSource, judgment: LLMScreeningJudgment) -> None:
    source.pre_llm_decision = source.pre_llm_decision or source.decision
    source.pre_llm_overall_score = source.pre_llm_overall_score or source.overall_score
    source.llm_judgment = judgment
    source.decision = judgment.decision
    source.overall_score = llm_blended_overall_score(source, judgment)
    source.reasons.append(
        "llm="
        f"{judgment.decision.value}:rel={judgment.agent_relevance:.2f},"
        f"reliability={judgment.reliability:.2f},"
        f"novelty={judgment.novelty:.2f},"
        f"classic={judgment.classic_value:.2f}"
    )
    if judgment.reasoning:
        source.reasons.append(f"llm_reason={judgment.reasoning}")


def llm_blended_overall_score(
    source: ScreenedSource,
    judgment: LLMScreeningJudgment,
) -> float:
    llm_score = (
        judgment.agent_relevance * 0.42
        + judgment.reliability * 0.25
        + judgment.novelty * 0.18
        + judgment.classic_value * 0.15
    )
    deterministic = source.pre_llm_overall_score or source.overall_score
    decision_bonus = {
        ScreeningDecision.ACCEPT: 0.04,
        ScreeningDecision.REVIEW: 0.0,
        ScreeningDecision.REJECT: -0.08,
    }[judgment.decision]
    return round(max(0.0, min(1.0, deterministic * 0.45 + llm_score * 0.55 + decision_bonus)), 3)


def recalculate_report_counts(report: ScreeningReport) -> None:
    report.total_candidates = len(report.sources)
    report.accepted = sum(1 for item in report.sources if item.decision == ScreeningDecision.ACCEPT)
    report.review = sum(1 for item in report.sources if item.decision == ScreeningDecision.REVIEW)
    report.rejected = sum(1 for item in report.sources if item.decision == ScreeningDecision.REJECT)
    report.selected_urls = [
        item.candidate.url for item in report.sources if item.decision == ScreeningDecision.ACCEPT
    ]


def normalize_llm_decision(value: object) -> str:
    text = str(value or "").lower().strip()
    if text not in {item.value for item in ScreeningDecision}:
        return ScreeningDecision.REVIEW.value
    return text


def normalize_memory_action(value: object) -> str:
    text = str(value or "").lower().strip()
    if text not in {"ingest", "watch", "skip"}:
        return "skip"
    return text


def normalize_concerns(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:120] for item in value[:5] if str(item).strip()]


def clamp_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 3)


def score_candidate(
    candidate: SourceCandidate,
    existing_texts: list[str] | None = None,
    min_accept_score: float = 0.5,
    min_relevance_score: float = 0.16,
) -> ScreenedSource:
    existing_texts = existing_texts or []
    relevance = score_agent_relevance(candidate)
    authority = score_authority(candidate)
    freshness = score_freshness(candidate.pushed_at)
    novelty, duplicate_of = score_novelty(candidate, existing_texts)
    scope = evaluate_discovery_scope(
        infer_discovery_source_type(candidate),
        published_at=candidate.pushed_at.date() if candidate.pushed_at else None,
        very_hot=is_very_hot_candidate(candidate),
    )
    overall = round(
        min(1.0, (relevance * 0.42) + (authority * 0.28) + (freshness * 0.2) + (novelty * 0.1)),
        3,
    )
    decision = decide_source(overall, relevance, novelty, min_accept_score, min_relevance_score)
    authority_override = authority >= 0.8 and relevance >= 0.14 and novelty >= 0.5
    if authority_override and scope.in_scope and overall >= min_accept_score * 0.9:
        decision = ScreeningDecision.ACCEPT
    if not scope.in_scope:
        decision = ScreeningDecision.REJECT
    reasons = build_reasons(candidate, relevance, authority, freshness, novelty, duplicate_of)
    if authority_override:
        reasons.append("authority_override_candidate")
    return ScreenedSource(
        candidate=candidate,
        decision=decision,
        overall_score=overall,
        relevance_score=relevance,
        authority_score=authority,
        freshness_score=freshness,
        novelty_score=novelty,
        duplicate_of=duplicate_of,
        reasons=reasons + [f"recency_status={scope.status}", f"recency_reason={scope.reason}"],
    )


def decide_source(
    overall: float,
    relevance: float,
    novelty: float,
    min_accept_score: float,
    min_relevance_score: float,
) -> ScreeningDecision:
    if relevance < min_relevance_score or novelty < 0.5:
        return ScreeningDecision.REJECT
    if overall >= min_accept_score:
        return ScreeningDecision.ACCEPT
    if overall >= min_accept_score * 0.75:
        return ScreeningDecision.REVIEW
    return ScreeningDecision.REJECT


def score_agent_relevance(candidate: SourceCandidate) -> float:
    primary_text = normalize_text(
        " ".join([candidate.title, candidate.summary, str(candidate.url)])
    )
    topic_text = normalize_text(" ".join(candidate.topics))
    core_hits = sum(1 for term in CORE_AGENT_TERMS if term in primary_text)
    supporting_hits = sum(1 for term in SUPPORTING_AGENT_TERMS if term in primary_text)
    topic_core_hits = sum(1 for term in CORE_AGENT_TERMS if term in topic_text)
    score = (core_hits * 0.14) + (supporting_hits * 0.04) + (topic_core_hits * 0.04)
    if core_hits == 0:
        score = min(score, 0.12)
    return round(min(1.0, score), 3)


def score_authority(candidate: SourceCandidate) -> float:
    stars = candidate.repo_stars or 0
    forks = candidate.repo_forks or 0
    followers = candidate.owner_followers or 0
    star_score = min(1.0, math.log10(stars + 1) / 4.0)
    fork_score = min(1.0, math.log10(forks + 1) / 3.2)
    follower_score = min(1.0, math.log10(followers + 1) / 4.0)
    metric_score = (star_score * 0.65) + (fork_score * 0.25) + (follower_score * 0.1)
    domain_score = authority_score_for_domain(candidate)
    return round(max(metric_score, domain_score), 3)


def authority_score_for_domain(candidate: SourceCandidate) -> float:
    domain = domain_from_url(str(candidate.url))
    if not domain:
        return 0.0
    if domain == "github.com" and candidate.repo_stars is None:
        return 0.45
    if domain in AUTHORITY_DOMAIN_SCORES:
        return AUTHORITY_DOMAIN_SCORES[domain]
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.62
    if domain.startswith("docs.") or domain.startswith("developer."):
        return 0.58
    return 0.0


def infer_discovery_source_type(candidate: SourceCandidate) -> str:
    domain = domain_from_url(str(candidate.url))
    if domain in AUTHORITY_DOMAIN_SCORES or domain.startswith("docs."):
        return "official_docs"
    if "spec" in candidate.title.lower() or "specification" in candidate.summary.lower():
        return "specification"
    if domain == "github.com":
        return "github_repository"
    return "article"


def is_very_hot_candidate(candidate: SourceCandidate) -> bool:
    return (candidate.repo_stars or 0) >= 10_000 or (candidate.repo_forks or 0) >= 2_000


def score_freshness(pushed_at: datetime | None) -> float:
    if pushed_at is None:
        return 0.35
    age_days = max(0, (datetime.now(UTC) - pushed_at.astimezone(UTC)).days)
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.85
    if age_days <= 180:
        return 0.65
    if age_days <= 730:
        return 0.4
    return 0.18


def score_novelty(
    candidate: SourceCandidate,
    existing_texts: list[str],
) -> tuple[float, str | None]:
    if not existing_texts:
        return 1.0, None
    candidate_summary = candidate_text(candidate)
    candidate_signature = distinctive_technical_terms(candidate_summary)
    matches = [
        score_memory_overlap(candidate_summary, candidate_signature, existing)
        for existing in existing_texts
    ]
    match = max(matches, key=lambda item: item["combined_similarity"])
    novelty = max(0.0, 1.0 - float(match["combined_similarity"]))
    new_terms = match["new_terms"]
    duplicate_of = None

    if match["is_duplicate"]:
        duplicate_of = str(match["existing"])
        novelty = min(novelty, 0.45)
    elif len(new_terms) >= 2:
        novelty = max(novelty, 0.55)

    return round(novelty, 3), duplicate_of


def score_memory_overlap(
    candidate_text_value: str,
    candidate_signature: set[str],
    existing_text_value: str,
) -> dict[str, object]:
    existing_signature = distinctive_technical_terms(existing_text_value)
    full_similarity = jaccard_similarity(candidate_text_value, existing_text_value)
    full_containment = token_containment_similarity(candidate_text_value, existing_text_value)
    signature_similarity = jaccard_set_similarity(candidate_signature, existing_signature)
    new_terms = candidate_signature - existing_signature

    if candidate_signature and existing_signature:
        combined_similarity = max(
            (signature_similarity * 0.65) + (full_similarity * 0.35),
            full_containment * 0.85,
        )
    else:
        combined_similarity = max(full_similarity, full_containment * 0.85)

    is_duplicate = (
        (signature_similarity >= 0.72 and len(new_terms) <= 1)
        or (full_similarity >= 0.88 and len(new_terms) <= 1)
        or (full_containment >= 0.78 and len(new_terms) <= 1)
    )
    return {
        "existing": existing_text_value,
        "full_similarity": full_similarity,
        "full_containment": full_containment,
        "signature_similarity": signature_similarity,
        "combined_similarity": combined_similarity,
        "new_terms": new_terms,
        "is_duplicate": is_duplicate,
    }


def build_reasons(
    candidate: SourceCandidate,
    relevance: float,
    authority: float,
    freshness: float,
    novelty: float,
    duplicate_of: str | None,
) -> list[str]:
    reasons = [
        f"relevance={relevance:.3f}",
        f"authority={authority:.3f}",
        f"freshness={freshness:.3f}",
        f"novelty={novelty:.3f}",
    ]
    if candidate.repo_stars is not None:
        reasons.append(f"stars={candidate.repo_stars}")
    if candidate.repo_forks is not None:
        reasons.append(f"forks={candidate.repo_forks}")
    if duplicate_of:
        reasons.append("similar_to_existing_memory")
        if authority >= 0.55:
            reasons.append("possible_better_rewrite_candidate")
    return reasons


def build_existing_memory_texts(index: KnowledgeIndex | None) -> list[str]:
    if index is None:
        return []
    return [
        normalize_text(
            " ".join([entry.card_title, entry.one_sentence, entry.agent_builder_takeaway])
        )
        for entry in index.entries
    ]


def load_knowledge_index(path: Path | None) -> KnowledgeIndex | None:
    if path is None or not path.exists():
        return None
    return KnowledgeIndex.model_validate_json(path.read_text(encoding="utf-8"))


def write_screening_report(report: ScreeningReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "source_screening.json", report.model_dump(mode="json"))
    (out_dir / "source_screening.md").write_text(render_screening_report(report), encoding="utf-8")
    (out_dir / "selected_urls.txt").write_text(
        "\n".join(str(url) for url in report.selected_urls) + "\n",
        encoding="utf-8",
    )


def render_screening_report(report: ScreeningReport) -> str:
    lines = [
        "# Source Screening Report",
        "",
        f"Total candidates: {report.total_candidates}",
        f"Accepted: {report.accepted}",
        f"Review: {report.review}",
        f"Rejected: {report.rejected}",
        f"LLM enabled: {str(report.llm_enabled).lower()}",
        f"LLM judged: {report.llm_judged}",
        "",
        "## Ranked Sources",
        "",
    ]
    for index, source in enumerate(report.sources, start=1):
        candidate = source.candidate
        lines.extend(
            [
                f"### {index}. {candidate.title}",
                "",
                f"URL: {candidate.url}",
                f"Decision: {source.decision.value}",
                f"Overall: {source.overall_score:.3f}",
                (
                    "Scores: "
                    f"relevance={source.relevance_score:.3f}, "
                    f"authority={source.authority_score:.3f}, "
                    f"freshness={source.freshness_score:.3f}, "
                    f"novelty={source.novelty_score:.3f}"
                ),
                f"Summary: {candidate.summary or '(none)'}",
                "Reasons: " + ", ".join(source.reasons),
            ]
        )
        if source.pre_llm_decision or source.llm_judgment:
            lines.extend(render_llm_judgment_lines(source))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_llm_judgment_lines(source: ScreenedSource) -> list[str]:
    lines: list[str] = []
    if source.pre_llm_decision:
        lines.append(f"Pre-LLM decision: {source.pre_llm_decision.value}")
    if source.pre_llm_overall_score is not None:
        lines.append(f"Pre-LLM overall: {source.pre_llm_overall_score:.3f}")
    if source.llm_judgment:
        judgment = source.llm_judgment
        lines.extend(
            [
                (
                    "LLM scores: "
                    f"agent_relevance={judgment.agent_relevance:.3f}, "
                    f"reliability={judgment.reliability:.3f}, "
                    f"novelty={judgment.novelty:.3f}, "
                    f"classic_value={judgment.classic_value:.3f}"
                ),
                f"LLM action: {judgment.memory_action}",
                f"LLM reason: {judgment.reasoning or '(none)'}",
            ]
        )
        if judgment.concerns:
            lines.append("LLM concerns: " + ", ".join(judgment.concerns))
    return lines


def candidate_text(candidate: SourceCandidate) -> str:
    return normalize_text(
        " ".join([candidate.title, candidate.summary, " ".join(candidate.topics)])
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", normalize_text(text))}


def distinctive_technical_terms(text: str) -> set[str]:
    return {
        token
        for token in token_set(text)
        if token not in NOVELTY_COMMON_TERMS
        and not token.isdigit()
        and len(token) >= 4
    }


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    return jaccard_set_similarity(left_tokens, right_tokens)


def token_containment_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def jaccard_set_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def parse_github_repo(url: str) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None, None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def first_text(*values: str | None) -> str:
    """Return the first non-empty metadata string."""
    for value in values:
        if value and value.strip():
            return re.sub(r"\s+", " ", value).strip()
    return ""


def meta_content(soup: BeautifulSoup, key: str) -> str | None:
    """Read common HTML metadata properties by name or property."""
    tag = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
    if not tag:
        return None
    content = tag.get("content")
    return str(content) if content else None


def first_datetime(*values: str | None) -> datetime | None:
    """Parse the first date-like metadata value."""
    for value in values:
        parsed = safe_parse_datetime(value)
        if parsed:
            return parsed
    return None


def safe_parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if len(text) == 10:
            text = f"{text}T00:00:00+00:00"
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def title_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return parsed.netloc or url
    return " ".join(part.replace("-", " ").replace("_", " ") for part in parts[-2:])


def topics_from_url(url: str) -> list[str]:
    parsed = urlparse(url)
    raw = [parsed.netloc, *parsed.path.split("/")]
    topics: list[str] = []
    for part in raw:
        for token in re.split(r"[^a-zA-Z0-9]+", part.lower()):
            if len(token) >= 3 and token not in {"www", "com", "html"}:
                topics.append(token)
    return unique_texts(topics)[:12]


def domain_from_url(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
