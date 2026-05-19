import json

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
) -> str:
    """Write a human report directly from the full knowledge index."""
    if not index.entries:
        return "# Frontier Agent Development Learning Report\n\nNo knowledge entries found.\n"

    result = await llm_client.chat_text(
        "validation",
        system_prompt=build_human_report_system_prompt(language=language),
        user_prompt=build_human_report_index_user_prompt(index, language=language),
        temperature=0.25,
    )
    markdown = normalize_markdown_report(str(result.content or ""), language=language)
    if not markdown:
        return "# Frontier Agent Development Learning Report\n\nNo report was generated.\n"
    if include_full_appendix:
        markdown = append_full_knowledge_appendix(markdown, index, language=language)
    return markdown.rstrip() + "\n"


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
        "Write a learning guide for a human learner, not for compact agent-memory "
        f"injection. {language_instruction}"
    )


def build_human_report_user_prompt(pack: AgentMemoryPack) -> str:
    entries = build_human_report_entries(pack)
    target_themes = 12 if len(entries) >= 70 else 10 if len(entries) >= 40 else 6
    target_chars = "30000-45000" if len(entries) >= 70 else "18000-30000"
    entries_json = json.dumps(entries, ensure_ascii=False)
    return (
        "Write a substantial English learning report for a human studying frontier "
        "agent development. The goal is a readable study guide, not a compact summary. "
        "Use the knowledge entries below as source-grounded material.\n"
        "Requirements:\n"
        "- Start with a short big-picture overview.\n"
        f"- Organize by about {target_themes} learning themes, not by raw entry order.\n"
        "- Cover memory, RAG/retrieval, knowledge graphs, orchestration, planning/reasoning, "
        "tool protocols, tool routing, tool schemas, stateful runtime, deployment, "
        "observability/evaluation, coding agents, browser/computer use, context engineering, "
        "model routing, cost/latency, identity/access, human-in-the-loop, guardrails, "
        "and safety when those topics appear in the entries.\n"
        "- For each major theme include: why it matters, key concepts, concrete "
        "implementation moves, common pitfalls, and 3-6 source URLs.\n"
        "- Add comparison tables where they help, especially for memory/RAG, orchestration, "
        "tooling, observability, and deployment choices.\n"
        "- Add a glossary of important terms, a final design checklist, and a short "
        "suggested reading path.\n"
        "- Include practical design checks, practice questions, and small implementation "
        "exercises.\n"
        "- Prefer source names and URLs over bare entry numbers. Entry IDs may be used "
        "as secondary references, but the reader should not need the raw memory pack.\n"
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


def build_human_report_index_user_prompt(index: KnowledgeIndex, language: str = "en") -> str:
    entries = build_human_report_index_entries(index)
    source_summary = build_source_summary(index)
    topic_summary = {topic.value: count for topic, count in index.topic_counts.items()}
    entries_json = json.dumps(entries, ensure_ascii=False)
    source_json = json.dumps(source_summary, ensure_ascii=False)
    topic_json = json.dumps(topic_summary, ensure_ascii=False)
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
        "Write the best possible human learning report from this full knowledge index.\n"
        f"{language_requirement}\n"
        "Audience: a developer learning frontier AI-agent engineering and preparing to "
        "build practical agent projects.\n"
        "Use all high-signal entries, but organize them into coherent learning themes.\n"
        "Requirements:\n"
        "- Start with an executive overview of the 2026 agent-engineering landscape.\n"
        "- Organize by 10-14 themes. Suggested themes include architecture, context "
        "engineering, memory/RAG, MCP/protocols, tool use and structured outputs, "
        "stateful runtime, multi-agent systems, observability/evaluation, security/"
        "guardrails, cost/latency, deployment, and human-in-the-loop.\n"
        "- For each theme include: core idea, why it matters, concrete implementation "
        "patterns, pitfalls, and source URLs.\n"
        "- Include at least three tables: theme map, implementation pattern matrix, "
        "and risk/mitigation checklist.\n"
        "- Include a practical architecture blueprint for a small but credible agent app.\n"
        "- Include a glossary, design checklist, reading path, and practice exercises.\n"
        "- Prefer implementation detail over vague summary. Use the provided "
        "implementation_notes whenever possible.\n"
        "- Do not invent facts beyond the provided entries. Attribute strong claims to "
        "specific source URLs.\n"
        "- Treat blogs and social posts as source-reported observations, not universal facts.\n"
        "- Preserve source URLs in Markdown links or plain URL form.\n"
        "- Target 35,000-60,000 characters if the content supports it; depth is more "
        "important than brevity.\n\n"
        f"Return Markdown beginning exactly with: {heading}\n\n"
        f"Generated at: {index.generated_at.isoformat()}\n"
        f"Total cards: {index.total_cards}\n"
        f"Topic summary: {topic_json}\n"
        f"Source summary: {source_json}\n"
        f"Knowledge entries: {entries_json}"
    )


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
    max_entries: int = 180,
    max_text_chars: int = 700,
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
                "evidence": [trim_text(item, max_text_chars) for item in entry.evidence[:2]],
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
    """Render every knowledge-index entry as a readable report appendix."""
    if not index.entries:
        return ""

    if language == "zh":
        lines = [
            "## 全量知识卡片附录",
            "",
            (
                "这一部分由程序根据当前 `knowledge_index.json` 稳定生成，"
                "用于保证 human report 覆盖全部知识卡片，而不是只保留 LLM 综述中提到的部分。"
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
            "evidence": "证据摘录",
        }
    else:
        lines = [
            "## Full Knowledge Card Appendix",
            "",
            (
                "This deterministic appendix is rendered from the current "
                "`knowledge_index.json` so the human report preserves all cards, not "
                "only the themes selected by the LLM narrative."
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
            "evidence": "Evidence",
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
        if entry.evidence:
            lines.extend(["", f"**{labels['evidence']}**:"])
            for evidence in entry.evidence[:2]:
                lines.append(f"- {evidence}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def trim_text(value: str, limit: int) -> str:
    """Keep prompt payload fields short without removing their semantic center."""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."
