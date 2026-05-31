import json
import re
from pathlib import Path
from typing import Any

from agent_knowledge_harvester.agents.blueprint import HUMAN_LEARNING_PROMPT
from agent_knowledge_harvester.memory.pack import render_human_learning_report
from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex
from agent_knowledge_harvester.schemas.memory import AgentMemoryPack


async def render_human_learning_report_with_llm(
    pack: AgentMemoryPack,
    llm_client: object,
) -> str:
    """Use the Human Learning Report Agent to write a readable English guide."""
    if not pack.entries:
        return render_human_learning_report(pack)

    result = await llm_client.chat_text(
        "validation",
        system_prompt=build_human_report_system_prompt(),
        user_prompt=build_human_report_user_prompt(pack),
        temperature=0.2,
    )
    markdown = normalize_markdown_report(str(result.content or ""))
    if not markdown:
        return render_human_learning_report(pack)
    return markdown.rstrip() + "\n"


async def render_human_learning_report_from_index_with_llm(
    index: KnowledgeIndex,
    llm_client: object,
    language: str = "en",
    include_full_appendix: bool = True,
    source_dossiers: list[dict[str, object]] | None = None,
    report_plan: str = "",
) -> str:
    """Write a human report directly from the full knowledge index."""
    if not index.entries:
        return "# Frontier Agent Development Learning Report\n\nNo knowledge entries found.\n"

    if not report_plan.strip():
        report_plan = await build_human_report_plan_with_llm(
            index,
            llm_client=llm_client,
            language=language,
            source_dossiers=source_dossiers,
        )
    result = await llm_client.chat_text(
        "validation",
        system_prompt=build_human_report_system_prompt(language=language),
        user_prompt=build_human_report_index_user_prompt(
            index,
            language=language,
            source_dossiers=source_dossiers,
            report_plan=report_plan,
        ),
        temperature=0.25,
    )
    markdown = normalize_markdown_report(str(result.content or ""), language=language)
    if not markdown:
        return "# Frontier Agent Development Learning Report\n\nNo report was generated.\n"
    markdown = insert_corpus_snapshot(markdown, index, language=language)
    if include_full_appendix:
        markdown = append_full_knowledge_appendix(markdown, index, language=language)
    return markdown.rstrip() + "\n"


async def build_human_report_plan_with_llm(
    index: KnowledgeIndex,
    llm_client: object,
    language: str = "en",
    source_dossiers: list[dict[str, object]] | None = None,
) -> str:
    """Ask the report planner to build a section-level research plan first."""
    result = await llm_client.chat_text(
        "validation",
        system_prompt=build_human_report_planner_system_prompt(language=language),
        user_prompt=build_human_report_planning_prompt(
            index,
            language=language,
            source_dossiers=source_dossiers,
        ),
        temperature=0.15,
    )
    return normalize_report_plan(str(result.content or ""))


def build_human_report_system_prompt(language: str = "en") -> str:
    language_instruction = (
        "Write in Simplified Chinese for a Chinese learner. Keep technical terms such as "
        "MCP, RAG, guardrails, structured outputs, stateful runtime, and observability "
        "in English with short Chinese explanations."
        if language == "zh"
        else "Write in clear English for a human learner."
    )
    return (
        HUMAN_LEARNING_PROMPT.strip()
        + "\n\nReturn Markdown only, with no JSON wrapper and no surrounding code fence. "
        "Write a polished enterprise/academic research briefing, not compact "
        "agent-memory injection and not a raw evidence dump. Prefer synthesized "
        "findings, decision implications, research gaps, and implementation patterns "
        "over card-by-card summaries. Cite source URLs for traceability, but do not "
        "paste raw evidence excerpts unless a short excerpt is essential to support a "
        "numeric, controversial, or source-specific claim. Use only provided image URLs "
        "or table snippets; never invent media, screenshots, diagrams, or table data. "
        f"{language_instruction}"
    )


def build_human_report_planner_system_prompt(language: str = "en") -> str:
    language_instruction = (
        "Write the plan in Simplified Chinese. Keep technical terms such as MCP, RAG, "
        "guardrails, stateful runtime, and observability in English when appropriate."
        if language == "zh"
        else "Write the plan in English."
    )
    return (
        "You are a senior research editor planning a rigorous enterprise/academic "
        "technical report. Return Markdown only. Do not write the final report yet. "
        "Your job is to design a section-level synthesis plan that will help a later "
        "writer produce detailed, paper-like prose grounded in the provided sources. "
        "Prefer problem framing, mechanisms, comparisons, limitations, and decision "
        f"implications over a list of source summaries. {language_instruction}"
    )


def build_human_report_user_prompt(pack: AgentMemoryPack) -> str:
    entries = build_human_report_entries(pack)
    target_themes = 12 if len(entries) >= 70 else 10 if len(entries) >= 40 else 6
    target_chars = "30000-45000" if len(entries) >= 70 else "18000-30000"
    entries_json = json.dumps(entries, ensure_ascii=False)
    return (
        "Write a substantial English research briefing for a human studying frontier "
        "agent development. The goal is an enterprise/academic investigation report, "
        "not a compact summary and not a list of evidence snippets. "
        "Use the knowledge entries below as source-grounded material.\n"
        "Requirements:\n"
        "- Start with an executive summary that states the 5-8 most decision-relevant "
        "findings.\n"
        "- Add a Methodology & Evidence Base section explaining that the report is "
        "synthesized from structured knowledge cards, with source URLs retained for "
        "traceability.\n"
        f"- Organize by about {target_themes} learning themes, not by raw entry order.\n"
        "- Cover memory, RAG/retrieval, knowledge graphs, orchestration, planning/reasoning, "
        "tool protocols, tool routing, tool schemas, stateful runtime, deployment, "
        "observability/evaluation, coding agents, browser/computer use, context engineering, "
        "model routing, cost/latency, identity/access, human-in-the-loop, guardrails, "
        "and safety when those topics appear in the entries.\n"
        "- For each major theme include: finding, why it matters, enterprise/academic "
        "interpretation, concrete implementation moves, common pitfalls, confidence "
        "level, and 3-6 source URLs.\n"
        "- Add comparison tables where they help, especially for memory/RAG, orchestration, "
        "tooling, observability, and deployment choices.\n"
        "- Add an adoption roadmap, research gaps/open questions, glossary of important "
        "terms, final design checklist, and a short suggested reading path.\n"
        "- Include practical design checks, practice questions, and small implementation "
        "exercises.\n"
        "- Prefer source names and URLs over bare entry numbers. Entry IDs may be used "
        "as secondary references, but the reader should not need the raw memory pack.\n"
        "- Use evidence fields only to verify synthesis. Do not create long evidence "
        "quote sections or bullet-dump excerpts.\n"
        "- If source entries include image URLs, include at most 3 remote images only "
        "when they are directly relevant to a finding. Caption each image with its "
        "source URL. Never invent image URLs.\n"
        "- If source entries include table snippets, do not paste them blindly. Rebuild "
        "only the useful rows as clean Markdown tables and cite the source URL.\n"
        "- Do not invent facts beyond the provided entries.\n"
        "- Attribute numeric claims and strong claims to a source; if a claim is from "
        "a blog or vendor article, phrase it as source-reported rather than universal fact.\n"
        "- Avoid absolute operational rules unless the source explicitly requires them.\n"
        "- Phrase safety guidance as design checks when the source only states a principle.\n"
        f"- Target roughly {target_chars} characters for large packs; it is acceptable "
        "for the report to be somewhat complex if the structure remains readable.\n"
        "- If a source entry includes image URLs, you may include up to 3 directly relevant "
        "remote images using Markdown image syntax, but never invent image URLs.\n\n"
        "Return Markdown beginning with: # Frontier Agent Development Learning Report\n\n"
        f"Generated at: {pack.generated_at.isoformat()}\n"
        f"Entries: {entries_json}"
    )


def build_human_report_index_user_prompt(
    index: KnowledgeIndex,
    language: str = "en",
    source_dossiers: list[dict[str, object]] | None = None,
    report_plan: str = "",
) -> str:
    entries = build_human_report_index_entries(index)
    source_summary = build_source_summary(index)
    corpus_facts = build_corpus_facts(index)
    topic_summary = {topic.value: count for topic, count in index.topic_counts.items()}
    entries_json = json.dumps(entries, ensure_ascii=False)
    source_json = json.dumps(source_summary, ensure_ascii=False)
    corpus_json = json.dumps(corpus_facts, ensure_ascii=False)
    topic_json = json.dumps(topic_summary, ensure_ascii=False)
    dossier_json = json.dumps(source_dossiers or [], ensure_ascii=False)
    plan_text = report_plan.strip()
    heading = (
        "# 前沿 Agent 开发学习报告"
        if language == "zh"
        else "# Frontier Agent Development Learning Report"
    )
    language_requirement = (
        "Write the report in Simplified Chinese. Use readable study-guide prose, not "
        "literal translation. Keep source URLs unchanged."
        if language == "zh"
        else "Write the report in English."
    )
    return (
        "Write the best possible enterprise/academic research report from this full "
        "knowledge index.\n"
        f"{language_requirement}\n"
        "Audience: technical leads, researchers, platform engineers, and developers who "
        "need to understand frontier AI-agent engineering and turn it into practical "
        "agent infrastructure decisions.\n"
        "A research editor has already prepared a section-level report plan. Follow "
        "that plan unless the evidence clearly requires a correction.\n"
        "Use all high-signal entries, but organize them into coherent learning themes.\n"
        "Requirements:\n"
        "- Start with an executive summary: 5-8 decision-relevant findings, each with "
        "a clear implication.\n"
        "- Add a Methodology & Evidence Base section explaining the corpus size, topic "
        "coverage, source mix, and confidence caveats. Do not pretend the corpus is a "
        "statistically complete literature review.\n"
        "- Use the authoritative corpus facts exactly as provided. Do not estimate, "
        "round, or invent counts for total cards, documents, unique sources, or topic "
        "coverage.\n"
        "- Organize the body by 10-14 research themes. Suggested themes include "
        "architecture, context "
        "engineering, memory/RAG, MCP/protocols, tool use and structured outputs, "
        "stateful runtime, multi-agent systems, observability/evaluation, security/"
        "guardrails, cost/latency, deployment, and human-in-the-loop.\n"
        "- For each theme include: finding, enterprise/academic interpretation, concrete "
        "implementation patterns, maturity/confidence, pitfalls, and source URLs.\n"
        "- Include at least four tables: executive finding map, implementation pattern "
        "matrix, evidence confidence matrix, and risk/mitigation checklist.\n"
        "- Include a practical architecture blueprint for a small but credible agent "
        "platform, plus an adoption roadmap from prototype to production.\n"
        "- Include research gaps/open questions and how the next discovery run should "
        "close them.\n"
        "- Include a glossary, design checklist, reading path, and practice exercises.\n"
        "- Prefer implementation detail over vague summary. Use the provided "
        "implementation_notes whenever possible.\n"
        "- Use the source_dossiers to add richer explanation, mechanism-level detail, "
        "methodology context, and paper-like connective prose. Knowledge cards provide "
        "the claim index; source_dossiers provide the longer source context. Do not "
        "ignore them when writing theme sections.\n"
        "- For important themes, write in research-report style: define the problem, "
        "summarize what multiple sources agree on, explain the mechanism, compare "
        "alternatives, then state implications and limitations.\n"
        "- Apply a depth contract to high-priority themes: each major theme should "
        "include 4-7 coherent paragraphs plus, when useful, one compact table or "
        "checklist. Cover background, mechanism, engineering design, trade-offs, "
        "failure modes, and what a team should do next. If the evidence is thin, say "
        "so and keep that section shorter.\n"
        "- For tooling/product themes such as Claude Code, Skills, OpenClaw, MCP "
        "servers, and coding-agent harnesses, explain what problem the tool solves, "
        "how it changes the agent workflow, where it fits in an enterprise stack, "
        "and what operational or security risks remain.\n"
        "- Use evidence fields as verification context, not as report filler. Avoid "
        "sections that merely paste evidence snippets. Summarize what the evidence "
        "supports and cite source URLs instead.\n"
        "- Use image_urls sparingly: include at most 3 source-provided remote images "
        "only when they materially improve explanation. Caption each image and cite "
        "the source URL. Never invent or download images.\n"
        "- Use table_snippets as source data for cleaner synthesis tables. Do not paste "
        "raw tables if they contain navigation, pricing noise, or irrelevant columns.\n"
        "- Do not invent facts beyond the provided entries. Attribute strong claims to "
        "specific source URLs.\n"
        "- Treat blogs and social posts as source-reported observations, not universal facts.\n"
        "- Preserve source URLs in Markdown links or plain URL form.\n"
        "- Target 35,000-60,000 characters if the content supports it; depth is more "
        "important than brevity.\n\n"
        f"Return Markdown beginning exactly with: {heading}\n\n"
        f"Generated at: {index.generated_at.isoformat()}\n"
        f"Total cards: {index.total_cards}\n"
        f"Authoritative corpus facts: {corpus_json}\n"
        f"Topic summary: {topic_json}\n"
        f"Source summary: {source_json}\n"
        f"Source dossiers for deep synthesis: {dossier_json}\n"
        f"Research report plan to follow: {plan_text}\n"
        f"Knowledge entries: {entries_json}"
    )


def build_human_report_planning_prompt(
    index: KnowledgeIndex,
    language: str = "en",
    source_dossiers: list[dict[str, object]] | None = None,
) -> str:
    """Build a compact planning prompt for the first report-writing stage."""
    entries = build_human_report_index_entries(index, max_entries=80, max_text_chars=420)
    source_summary = build_source_summary(index)[:40]
    corpus_facts = build_corpus_facts(index)
    topic_summary = {topic.value: count for topic, count in index.topic_counts.items()}
    heading = "研究报告写作计划" if language == "zh" else "Research Report Writing Plan"
    language_requirement = (
        "Use Simplified Chinese." if language == "zh" else "Use English."
    )
    return (
        f"Create a Markdown {heading} for a detailed enterprise/academic report about "
        "frontier AI-agent engineering.\n"
        f"{language_requirement}\n"
        "The final writer will receive the full card index; this planning stage should "
        "decide how to organize the argument and where to add depth.\n"
        "Plan requirements:\n"
        "- Propose 10-14 major sections with concise titles.\n"
        "- For each major section specify: problem framing, mechanism to explain, "
        "sources to cite, comparison table ideas, limitations/caveats, and concrete "
        "engineering implications.\n"
        "- Mark which sections deserve deep treatment and list the 4-6 subquestions "
        "the writer should answer there: background, mechanism, implementation path, "
        "trade-offs, failure modes, and follow-up research.\n"
        "- Reserve explicit slots for any evidence-backed tooling themes such as "
        "Claude Code, Skills, OpenClaw, MCP servers, and coding-agent harnesses.\n"
        "- Identify 5-8 executive-summary findings and the evidence behind each.\n"
        "- Identify which source dossiers should be used for deep discussion.\n"
        "- Identify where to include architecture blueprint, adoption roadmap, "
        "risk/mitigation checklist, research gaps, glossary, and exercises.\n"
        "- Mark any numeric or strong claims as source-reported unless backed by "
        "official docs or papers.\n"
        "- Do not invent sections outside the evidence base.\n\n"
        f"Authoritative corpus facts: {json.dumps(corpus_facts, ensure_ascii=False)}\n"
        f"Topic summary: {json.dumps(topic_summary, ensure_ascii=False)}\n"
        f"Source summary: {json.dumps(source_summary, ensure_ascii=False)}\n"
        f"Source dossiers: {json.dumps(source_dossiers or [], ensure_ascii=False)}\n"
        f"Representative knowledge entries: {json.dumps(entries, ensure_ascii=False)}"
    )


def normalize_report_plan(content: str) -> str:
    """Remove wrappers and bound the planning handoff."""
    plan = content.strip()
    if plan.startswith("```"):
        lines = plan.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        plan = "\n".join(lines).strip()
    return trim_markdown(plan, 16_000)


def trim_markdown(value: str, limit: int) -> str:
    """Trim Markdown while preserving useful line breaks."""
    normalized = re.sub(r"\n{4,}", "\n\n\n", value.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def normalize_markdown_report(content: str, language: str = "en") -> str:
    """Remove common model wrappers while preserving the report body."""
    markdown = content.strip()
    if markdown.startswith("```"):
        lines = markdown.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        markdown = "\n".join(lines).strip()
    expected_heading = (
        "# 前沿 Agent 开发学习报告"
        if language == "zh"
        else "# Frontier Agent Development Learning Report"
    )
    heading_index = markdown.find(expected_heading)
    if heading_index > 0:
        markdown = markdown[heading_index:].strip()
    return markdown


def build_human_report_entries(
    pack: AgentMemoryPack,
    max_entries: int = 140,
    max_text_chars: int = 520,
) -> list[dict[str, object]]:
    """Prepare a bounded but information-rich payload for the report-writing expert."""
    entries = []
    for index, entry in enumerate(pack.entries[:max_entries], start=1):
        entries.append(
            {
                "id": index,
                "title": trim_text(entry.card_title, max_text_chars),
                "claim": trim_text(entry.claim, max_text_chars),
                "agent_move": trim_text(entry.agent_move, max_text_chars),
                "topics": [topic.value for topic in entry.topics],
                "source_url": str(entry.source_url),
                "evidence": [trim_text(item, max_text_chars) for item in entry.evidence[:2]],
                "priority_score": entry.priority_score,
            }
        )
    return entries


def build_human_report_index_entries(
    index: KnowledgeIndex,
    max_entries: int = 220,
    max_text_chars: int = 850,
) -> list[dict[str, object]]:
    """Prepare rich knowledge-index entries for human report generation."""
    entries = []
    for position, entry in enumerate(index.entries[:max_entries], start=1):
        entries.append(
            {
                "id": position,
                "title": trim_text(entry.card_title, max_text_chars),
                "source_title": trim_text(entry.source_title or "", max_text_chars),
                "source_url": str(entry.source_url),
                "claim": trim_text(entry.one_sentence, max_text_chars),
                "why_it_matters": trim_text(entry.why_it_matters, max_text_chars),
                "agent_move": trim_text(entry.agent_builder_takeaway, max_text_chars),
                "implementation_notes": [
                    trim_text(item, max_text_chars) for item in entry.implementation_notes[:4]
                ],
                "topics": [topic.value for topic in entry.topics],
                "verification_excerpts_for_synthesis_only": [
                    trim_text(item, max_text_chars) for item in entry.evidence[:2]
                ],
                "image_urls": entry.image_urls[:3],
                "table_snippets": [
                    trim_text(item, max_text_chars) for item in entry.table_snippets[:2]
                ],
                "priority_score": entry.priority_score,
                "relevance_score": entry.relevance_score,
                "frontier_score": entry.frontier_score,
            }
        )
    return entries


def build_source_summary(index: KnowledgeIndex) -> list[dict[str, object]]:
    """Summarize sources so the report can cite recurring authorities."""
    source_counts: dict[str, int] = {}
    source_titles: dict[str, str] = {}
    for entry in index.entries:
        source = str(entry.source_url)
        source_counts[source] = source_counts.get(source, 0) + 1
        if entry.source_title:
            source_titles[source] = entry.source_title
    return [
        {
            "source_url": source,
            "source_title": source_titles.get(source, ""),
            "cards": count,
        }
        for source, count in sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
    ]


def build_source_dossiers_from_ingestion_dir(
    index: KnowledgeIndex,
    ingestion_dir: Path,
    max_sources: int = 36,
    max_excerpt_chars: int = 2800,
) -> list[dict[str, object]]:
    """Build longer source notes for report writing from already-ingested documents."""
    if not ingestion_dir.exists():
        return []

    source_rank = rank_sources_for_dossiers(index)
    if not source_rank:
        return []

    ingested = load_ingested_documents_by_url(ingestion_dir)
    dossiers: list[dict[str, object]] = []
    for source_url, rank in source_rank[:max_sources]:
        document = ingested.get(source_url)
        if not document:
            continue
        markdown = str(document.get("markdown") or "")
        if not markdown.strip():
            continue
        dossier = {
            "source_url": source_url,
            "source_title": document.get("title") or rank["source_title"],
            "card_count": rank["card_count"],
            "max_priority": rank["max_priority"],
            "topics": rank["topics"],
            "top_card_titles": rank["top_card_titles"],
            "summary_hint": trim_text(str(document.get("summary_hint") or ""), 500),
            "source_outline": extract_source_outline(markdown),
            "selected_passages": extract_research_passages(markdown, max_excerpt_chars),
        }
        dossiers.append(dossier)
    return dossiers


def rank_sources_for_dossiers(index: KnowledgeIndex) -> list[tuple[str, dict[str, Any]]]:
    """Rank sources by card density and priority so report context stays bounded."""
    by_source: dict[str, dict[str, Any]] = {}
    for entry in index.entries:
        source = str(entry.source_url)
        item = by_source.setdefault(
            source,
            {
                "source_title": entry.source_title or "",
                "card_count": 0,
                "max_priority": 0.0,
                "priority_sum": 0.0,
                "topics": set(),
                "top_card_titles": [],
            },
        )
        item["card_count"] += 1
        item["max_priority"] = max(item["max_priority"], entry.priority_score)
        item["priority_sum"] += entry.priority_score
        item["topics"].update(topic.value for topic in entry.topics)
        if len(item["top_card_titles"]) < 5:
            item["top_card_titles"].append(entry.card_title)
        if entry.source_title and not item["source_title"]:
            item["source_title"] = entry.source_title

    ranked: list[tuple[str, dict[str, Any]]] = []
    for source, item in by_source.items():
        item["topics"] = sorted(item["topics"])
        ranked.append((source, item))
    return sorted(
        ranked,
        key=lambda pair: (
            pair[1]["max_priority"],
            pair[1]["priority_sum"],
            pair[1]["card_count"],
        ),
        reverse=True,
    )


def load_ingested_documents_by_url(ingestion_dir: Path) -> dict[str, dict[str, object]]:
    """Load successful ingestion artifacts keyed by clean source URL."""
    documents: dict[str, dict[str, object]] = {}
    for path in sorted(ingestion_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not payload.get("success"):
            continue
        clean = payload.get("clean")
        if not isinstance(clean, dict):
            continue
        source_url = str(clean.get("source_url") or "")
        markdown = str(clean.get("markdown") or "")
        if not source_url or not markdown.strip():
            continue
        documents[source_url] = {
            "source_url": source_url,
            "title": clean.get("title") or "",
            "markdown": markdown,
            "summary_hint": clean.get("summary_hint") or "",
        }
    return documents


def extract_source_outline(markdown: str, max_headings: int = 18) -> list[str]:
    """Extract a compact heading outline for report-level source context."""
    outline: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}(#{1,4})\s+(.+?)\s*$", line)
        if not match:
            continue
        heading = strip_inline_markdown(match.group(2))
        if is_low_value_report_line(heading):
            continue
        outline.append(heading)
        if len(outline) >= max_headings:
            break
    return outline


def extract_research_passages(markdown: str, max_chars: int) -> str:
    """Select source passages that help long-form research synthesis."""
    paragraphs = split_report_paragraphs(markdown)
    selected: list[str] = []

    for paragraph in paragraphs[:8]:
        if is_useful_research_passage(paragraph):
            selected.append(paragraph)
            break

    for paragraph in paragraphs:
        if not is_useful_research_passage(paragraph):
            continue
        if paragraph in selected:
            continue
        selected.append(paragraph)
        if sum(len(item) + 2 for item in selected) >= max_chars:
            break

    if not selected:
        selected = paragraphs[:3]
    return trim_text("\n\n".join(selected), max_chars)


def split_report_paragraphs(markdown: str) -> list[str]:
    """Split markdown into clean paragraphs suitable for LLM report context."""
    blocks = re.split(r"\n\s*\n", markdown)
    paragraphs: list[str] = []
    for block in blocks:
        text = strip_inline_markdown(block)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 120 or is_low_value_report_line(text):
            continue
        paragraphs.append(text)
    return paragraphs


def is_useful_research_passage(text: str) -> bool:
    """Prefer mechanism, evaluation, architecture, and limitation passages."""
    lowered = text.lower()
    keywords = [
        "architecture",
        "benchmark",
        "case study",
        "comparison",
        "cost",
        "design",
        "evaluation",
        "experiment",
        "framework",
        "guardrail",
        "implementation",
        "latency",
        "limitation",
        "memory",
        "method",
        "mcp",
        "observability",
        "protocol",
        "retrieval",
        "risk",
        "runtime",
        "security",
        "state",
        "tool",
        "workflow",
    ]
    return any(keyword in lowered for keyword in keywords)


def is_low_value_report_line(text: str) -> bool:
    """Filter common navigation and repository chrome from report dossiers."""
    lowered = text.strip().lower()
    if not lowered:
        return True
    noise = [
        "skip to content",
        "navigation menu",
        "sign in",
        "sign up",
        "appearance settings",
        "github copilot",
        "star history",
        "last commit",
        "contributors",
        "license",
        "fork",
        "stars",
    ]
    return any(item in lowered for item in noise)


def strip_inline_markdown(text: str) -> str:
    """Remove lightweight markdown markup while preserving readable text."""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("*", "")
    text = text.replace("_", "")
    text = text.replace("|", " ")
    return text.strip()


def build_corpus_facts(index: KnowledgeIndex) -> dict[str, object]:
    """Return deterministic report-level facts that should not be left to the LLM."""
    unique_sources = sorted({str(entry.source_url) for entry in index.entries})
    topic_counts = {topic.value: count for topic, count in index.topic_counts.items()}
    if not topic_counts:
        for entry in index.entries:
            for topic in entry.topics:
                topic_counts[topic.value] = topic_counts.get(topic.value, 0) + 1
    return {
        "generated_at": index.generated_at.isoformat(),
        "total_documents": index.total_documents,
        "total_cards": index.total_cards,
        "unique_sources": len(unique_sources),
        "topic_count": len(topic_counts),
        "top_topics": [
            {"topic": topic, "cards": count}
            for topic, count in sorted(
                topic_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        ],
    }


def insert_corpus_snapshot(
    markdown: str,
    index: KnowledgeIndex,
    language: str = "en",
) -> str:
    """Insert a deterministic report snapshot after the heading."""
    if "## Report Snapshot" in markdown or "## 报告快照" in markdown:
        return markdown

    snapshot = render_corpus_snapshot(index, language=language)
    if not snapshot:
        return markdown

    lines = markdown.splitlines()
    if not lines:
        return snapshot

    insert_at = 1 if lines[0].startswith("# ") else 0
    return "\n".join(lines[:insert_at] + ["", snapshot, ""] + lines[insert_at:]).strip()


def render_corpus_snapshot(index: KnowledgeIndex, language: str = "en") -> str:
    """Render deterministic corpus statistics for the human report."""
    facts = build_corpus_facts(index)
    top_topics = facts["top_topics"]
    if language == "zh":
        lines = [
            "## 报告快照",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 生成时间 | {facts['generated_at']} |",
            f"| 文档数 | {facts['total_documents']} |",
            f"| 知识卡片数 | {facts['total_cards']} |",
            f"| 唯一来源 URL 数 | {facts['unique_sources']} |",
            f"| 覆盖主题数 | {facts['topic_count']} |",
        ]
        if top_topics:
            lines.extend(["", "**高频主题（Top 12）**："])
            lines.extend(
                f"- {item['topic']}: {item['cards']} cards" for item in top_topics
            )
    else:
        lines = [
            "## Report Snapshot",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Generated at | {facts['generated_at']} |",
            f"| Documents | {facts['total_documents']} |",
            f"| Knowledge cards | {facts['total_cards']} |",
            f"| Unique source URLs | {facts['unique_sources']} |",
            f"| Covered topics | {facts['topic_count']} |",
        ]
        if top_topics:
            lines.extend(["", "**Top topics (12):**"])
            lines.extend(
                f"- {item['topic']}: {item['cards']} cards" for item in top_topics
            )
    return "\n".join(lines)


def append_full_knowledge_appendix(
    markdown: str,
    index: KnowledgeIndex,
    language: str = "en",
) -> str:
    """Append a deterministic all-card appendix so human reports keep full coverage."""
    appendix = render_full_knowledge_appendix(index, language=language)
    if not appendix:
        return markdown
    return markdown.rstrip() + "\n\n" + appendix


def render_full_knowledge_appendix(index: KnowledgeIndex, language: str = "en") -> str:
    """Render every knowledge-index entry as a compact source register appendix."""
    if not index.entries:
        return ""

    if language == "zh":
        lines = [
            "## 全量来源登记附录",
            "",
            (
                "这一部分由程序根据当前 `knowledge_index.json` 稳定生成，"
                "用于保证 human report 覆盖全部知识卡片。这里保留来源、主题、核心观点和"
                "工程启发，不再默认展开原始 evidence 摘录，避免把调研报告变成证据堆砌。"
            ),
            "",
            f"- 总文档数：{index.total_documents}",
            f"- 总知识卡片数：{index.total_cards}",
            "",
        ]
        labels = {
            "source": "来源",
            "source_title": "来源标题",
            "topics": "主题",
            "score": "优先级",
            "core": "核心观点",
            "why": "为什么重要",
            "takeaway": "Agent 构建启发",
            "implementation": "实现要点",
        }
    else:
        lines = [
            "## Full Source Register Appendix",
            "",
            (
                "This deterministic appendix is rendered from the current "
                "`knowledge_index.json` so the human report preserves all cards. It "
                "keeps source, topic, claim, and implementation traceability without "
                "turning the report into a raw evidence dump."
            ),
            "",
            f"- Total documents: {index.total_documents}",
            f"- Total knowledge cards: {index.total_cards}",
            "",
        ]
        labels = {
            "source": "Source",
            "source_title": "Source title",
            "topics": "Topics",
            "score": "Priority",
            "core": "Core idea",
            "why": "Why it matters",
            "takeaway": "Agent builder takeaway",
            "implementation": "Implementation details",
        }

    for position, entry in enumerate(index.entries, start=1):
        topics = ", ".join(topic.value for topic in entry.topics) or "uncategorized"
        lines.extend(
            [
                f"### A{position}. {entry.card_title}",
                "",
                f"- {labels['source']}: {entry.source_url}",
                f"- {labels['topics']}: {topics}",
                (
                    f"- {labels['score']}: {entry.priority_score:.3f} "
                    f"(relevance={entry.relevance_score:.3f}, frontier={entry.frontier_score:.3f})"
                ),
            ]
        )
        if entry.source_title:
            lines.append(f"- {labels['source_title']}: {entry.source_title}")

        lines.extend(
            [
                "",
                f"**{labels['core']}**: {entry.one_sentence}",
            ]
        )
        if entry.why_it_matters:
            lines.append(f"**{labels['why']}**: {entry.why_it_matters}")
        lines.append(f"**{labels['takeaway']}**: {entry.agent_builder_takeaway}")

        if entry.implementation_notes:
            lines.extend(["", f"**{labels['implementation']}**:"])
            for note in entry.implementation_notes:
                lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def trim_text(value: str, limit: int) -> str:
    """Keep prompt payload fields short without removing their semantic center."""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."
