import json

from agent_knowledge_harvester.agents.blueprint import HUMAN_LEARNING_PROMPT
from agent_knowledge_harvester.memory.pack import render_human_learning_report
from agent_knowledge_harvester.schemas.memory import AgentMemoryPack


async def render_human_learning_report_with_llm(
    pack: AgentMemoryPack,
    llm_client: object,
) -> str:
    """Use the Human Learning Report Agent to write a readable English guide."""
    if not pack.entries:
        return render_human_learning_report(pack)

    result = await llm_client.chat_json(
        "validation",
        system_prompt=build_human_report_system_prompt(),
        user_prompt=build_human_report_user_prompt(pack),
        temperature=0.2,
    )
    markdown = str(result.payload.get("markdown") or "").strip()
    if not markdown:
        return render_human_learning_report(pack)
    return markdown.rstrip() + "\n"


def build_human_report_system_prompt() -> str:
    return (
        HUMAN_LEARNING_PROMPT.strip()
        + "\n\nReturn strict JSON only with one key: markdown. "
        "The markdown value must be an English learning guide. Write for a human "
        "learner, not for compact agent-memory injection."
    )


def build_human_report_user_prompt(pack: AgentMemoryPack) -> str:
    entries = build_human_report_entries(pack)
    target_themes = 8 if len(entries) >= 50 else 6 if len(entries) >= 20 else 4
    target_chars = "18000-28000" if len(entries) >= 50 else "10000-18000"
    entries_json = json.dumps(entries, ensure_ascii=False)
    return (
        "Write a substantial English learning report for a human studying frontier "
        "agent development. The goal is a readable study guide, not a compact summary. "
        "Use the knowledge entries below as source-grounded material.\n"
        "Requirements:\n"
        "- Start with a short big-picture overview.\n"
        f"- Organize by about {target_themes} learning themes, not by raw entry order.\n"
        "- Cover memory, orchestration, tool protocols, tool schemas, deployment, "
        "observability/evaluation, coding agents, context engineering, and safety "
        "when those topics appear in the entries.\n"
        "- For each major theme include: why it matters, key concepts, concrete "
        "implementation moves, common pitfalls, and 2-4 source URLs.\n"
        "- Add a glossary of important terms and a final design checklist.\n"
        "- Include practical design checks or practice questions.\n"
        "- Prefer source names and URLs over bare entry numbers. Entry IDs may be used "
        "as secondary references, but the reader should not need the raw memory pack.\n"
        "- Do not invent facts beyond the provided entries.\n"
        "- Attribute numeric claims and strong claims to a source; if a claim is from "
        "a blog or vendor article, phrase it as source-reported rather than universal fact.\n"
        "- Avoid absolute operational rules unless the source explicitly requires them.\n"
        "- Phrase safety guidance as design checks when the source only states a principle.\n"
        f"- Target roughly {target_chars} characters for large packs; do not collapse "
        "many distinct cards into four short themes.\n\n"
        "Return JSON: {\"markdown\": \"# Frontier Agent Development Learning Report\\n...\"}\n\n"
        f"Generated at: {pack.generated_at.isoformat()}\n"
        f"Entries: {entries_json}"
    )


def build_human_report_entries(
    pack: AgentMemoryPack,
    max_entries: int = 90,
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


def trim_text(value: str, limit: int) -> str:
    """Keep prompt payload fields short without removing their semantic center."""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."
