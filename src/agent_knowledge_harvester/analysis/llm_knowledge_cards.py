import asyncio
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_knowledge_harvester.agents.blueprint import DEEP_READER_PROMPT
from agent_knowledge_harvester.analysis.knowledge_cards import (
    KnowledgeCardAnalyzer,
    build_frontier_brief,
    build_knowledge_index,
    clamp_text,
    detect_topics,
    iter_ingestion_json_files,
    load_ingestion_result,
    render_frontier_brief,
    render_index_markdown,
    render_markdown,
)
from agent_knowledge_harvester.schemas.analysis import (
    AnalysisResult,
    AnalysisRunStats,
    KnowledgeCard,
    KnowledgeIndex,
    KnowledgeTopic,
)
from agent_knowledge_harvester.schemas.ingestion import CleanDocument, IngestionResult
from agent_knowledge_harvester.utils.files import stable_slug, write_json


class LLMKnowledgeCardAnalyzer:
    """Use the Deep Reading Agent prompt to extract source-grounded cards."""

    def __init__(
        self,
        llm_client: object,
        max_cards_per_doc: int = 4,
        max_markdown_chars: int = 18_000,
        fallback_analyzer: KnowledgeCardAnalyzer | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_cards_per_doc = max_cards_per_doc
        self.max_markdown_chars = max_markdown_chars
        self.fallback_analyzer = fallback_analyzer or KnowledgeCardAnalyzer(
            max_cards_per_doc=max_cards_per_doc
        )
        self.llm_calls = 0
        self.prompt_tokens_estimate = 0
        self.completion_tokens_estimate = 0
        self.fallback_count = 0
        self.fallback_errors: list[str] = []

    async def analyze_ingestion_result(self, result: IngestionResult) -> AnalysisResult | None:
        if not result.success or result.clean is None:
            return None
        try:
            return await self.analyze_document(result.clean)
        except Exception as exc:
            self.fallback_count += 1
            self.fallback_errors.append(type(exc).__name__ + ": " + str(exc)[:180])
            fallback = self.fallback_analyzer.analyze_document(result.clean)
            if not fallback.cards:
                fallback.skipped_reason = "llm_extraction_failed_fallback_used"
            return fallback

    async def analyze_document(self, document: CleanDocument) -> AnalysisResult:
        llm_result = await self.llm_client.chat_json(
            "extraction",
            system_prompt=build_llm_extraction_system_prompt(),
            user_prompt=build_llm_extraction_user_prompt(
                document,
                max_cards=self.max_cards_per_doc,
                max_markdown_chars=self.max_markdown_chars,
            ),
            temperature=0.0,
        )
        self.llm_calls += 1
        self.prompt_tokens_estimate += llm_result.prompt_tokens_estimate
        self.completion_tokens_estimate += llm_result.completion_tokens_estimate
        return parse_llm_analysis_result(
            payload=llm_result.payload,
            document=document,
            max_cards=self.max_cards_per_doc,
        )


async def analyze_directory_with_llm(
    in_dir: Path,
    out_dir: Path,
    analyzer: LLMKnowledgeCardAnalyzer,
    max_index_entries: int = 30,
    concurrency: int = 1,
) -> tuple[list[AnalysisResult], AnalysisRunStats, KnowledgeIndex]:
    """Analyze ingestion outputs with the LLM deep-reader and write standard artifacts."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def analyze_path(path: Path) -> AnalysisResult | None:
        result = load_ingestion_result(path)
        if result is None:
            return None
        async with semaphore:
            return await analyzer.analyze_ingestion_result(result)

    analyses = await asyncio.gather(
        *(analyze_path(path) for path in iter_ingestion_json_files(in_dir))
    )
    results = [analysis for analysis in analyses if analysis is not None]

    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        slug = stable_slug(str(result.source_url))
        write_json(out_dir / f"{slug}.knowledge.json", result.model_dump(mode="json"))
        (out_dir / f"{slug}.knowledge.md").write_text(render_markdown(result), encoding="utf-8")

    stats = AnalysisRunStats().finish(results)
    write_json(out_dir / "analysis_stats.json", stats.model_dump(mode="json"))
    index = build_knowledge_index(results, max_entries=max_index_entries)
    write_json(out_dir / "knowledge_index.json", index.model_dump(mode="json"))
    (out_dir / "knowledge_index.md").write_text(render_index_markdown(index), encoding="utf-8")
    brief = build_frontier_brief(index)
    write_json(out_dir / "frontier_brief.json", brief.model_dump(mode="json"))
    (out_dir / "frontier_brief.md").write_text(render_frontier_brief(brief), encoding="utf-8")
    return results, stats, index


def build_llm_extraction_system_prompt() -> str:
    return (
        DEEP_READER_PROMPT.strip()
        + "\n\nReturn strict JSON only. Never include markdown fences or commentary."
    )


def build_llm_extraction_user_prompt(
    document: CleanDocument,
    max_cards: int,
    max_markdown_chars: int,
) -> str:
    topics = [topic.value for topic in KnowledgeTopic]
    markdown = clamp_text(document.markdown, max_markdown_chars)
    return (
        "Extract source-grounded agent-development knowledge cards from this document.\n"
        f"Source URL: {document.source_url}\n"
        f"Title: {document.title or '(none)'}\n"
        f"Summary hint: {document.summary_hint}\n"
        f"Max cards: {max_cards}\n"
        f"Allowed topics: {topics}\n"
        "If the source introduces a useful concept that does not fit the allowed topics, "
        "choose the closest allowed topic and mention the new concept explicitly in the "
        "title, claim, or implementation notes.\n"
        "Each card must be atomic, specific, and useful for building agents.\n"
        "Use short evidence snippets copied or tightly paraphrased from the source.\n"
        "Avoid installation steps, marketing, navigation, file listings, and generic claims.\n"
        "JSON schema:\n"
        "{\n"
        '  "cards": [\n'
        "    {\n"
        '      "title": "short card title",\n'
        '      "one_sentence": "core claim under 40 words",\n'
        '      "why_it_matters": "why agent builders should care",\n'
        '      "agent_builder_takeaway": "specific implementation/design move",\n'
        '      "topics": ["mcp"],\n'
        '      "implementation_notes": ["short note"],\n'
        '      "evidence": ["short source-grounded snippet"],\n'
        '      "relevance_score": 0.0,\n'
        '      "frontier_score": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "skipped_reason": null\n'
        "}\n\n"
        "Document markdown:\n"
        f"{markdown}"
    )


def parse_llm_analysis_result(
    payload: dict[str, Any],
    document: CleanDocument,
    max_cards: int,
) -> AnalysisResult:
    raw_cards = payload.get("cards", [])
    if not isinstance(raw_cards, list):
        raw_cards = []
    cards: list[KnowledgeCard] = []
    for raw_card in raw_cards[:max_cards]:
        if not isinstance(raw_card, dict):
            continue
        card = parse_llm_card(raw_card, document)
        if card is not None:
            cards.append(card)

    skipped_reason = None
    if not cards:
        skipped_reason = str(payload.get("skipped_reason") or "llm returned no useful cards")

    return AnalysisResult(
        source_url=document.source_url,
        title=document.title,
        cards=cards,
        skipped_reason=skipped_reason,
    )


def parse_llm_card(raw_card: dict[str, Any], document: CleanDocument) -> KnowledgeCard | None:
    text_for_topics = " ".join(
        str(raw_card.get(key, ""))
        for key in ("title", "one_sentence", "why_it_matters", "agent_builder_takeaway")
    )
    topics = parse_topics(raw_card.get("topics"), fallback_text=text_for_topics)
    normalized = {
        "source_url": document.source_url,
        "title": clamp_text(str(raw_card.get("title") or document.title or "Untitled"), 90),
        "one_sentence": clamp_text(str(raw_card.get("one_sentence") or ""), 260),
        "why_it_matters": clamp_text(str(raw_card.get("why_it_matters") or ""), 260),
        "agent_builder_takeaway": clamp_text(
            str(raw_card.get("agent_builder_takeaway") or ""),
            260,
        ),
        "topics": topics,
        "implementation_notes": clamp_list(raw_card.get("implementation_notes"), 4, 220),
        "evidence": clamp_list(raw_card.get("evidence"), 3, 220),
        "relevance_score": clamp_score(raw_card.get("relevance_score"), default=0.5),
        "frontier_score": clamp_score(raw_card.get("frontier_score"), default=0.3),
    }
    if not normalized["one_sentence"] or not normalized["agent_builder_takeaway"]:
        return None
    try:
        return KnowledgeCard.model_validate(normalized)
    except ValidationError:
        return None


def parse_topics(value: object, fallback_text: str) -> list[KnowledgeTopic]:
    topics: list[KnowledgeTopic] = []
    if isinstance(value, list):
        for item in value:
            try:
                topics.append(KnowledgeTopic(str(item).strip()))
            except ValueError:
                continue
    return topics or detect_topics(fallback_text)


def clamp_list(value: object, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        clamp_text(str(item), max_chars)
        for item in value[:limit]
        if str(item).strip()
    ]


def clamp_score(value: object, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return round(max(0.0, min(1.0, score)), 3)
