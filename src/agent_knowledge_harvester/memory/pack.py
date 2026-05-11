import re
from datetime import UTC, datetime
from pathlib import Path

from agent_knowledge_harvester.analysis.source_screening import jaccard_similarity
from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex, KnowledgeIndexEntry
from agent_knowledge_harvester.schemas.memory import (
    AgentMemoryEntry,
    AgentMemoryPack,
    CompactMemoryPack,
)
from agent_knowledge_harvester.utils.files import write_json


def build_memory_pack(
    index_paths: list[Path],
    after: datetime | None = None,
    min_priority: float = 0.0,
    max_entries: int | None = 30,
    dedupe_threshold: float | None = 0.82,
) -> AgentMemoryPack:
    """Build a time-filterable memory pack from knowledge indexes."""
    pack = AgentMemoryPack(
        source_indexes=[str(path) for path in index_paths],
        after=after,
    )
    entries: list[AgentMemoryEntry] = []
    seen_texts: list[str] = []

    for index_path in index_paths:
        index = load_knowledge_index(index_path)
        for item in index.entries:
            pack.total_input_entries += 1
            if after and index.generated_at <= after:
                pack.dropped_by_time += 1
                continue
            if item.priority_score < min_priority:
                pack.dropped_by_priority += 1
                continue

            fingerprint = memory_fingerprint(item)
            if dedupe_threshold is not None and is_duplicate_memory(
                fingerprint,
                seen_texts,
                threshold=dedupe_threshold,
            ):
                pack.dropped_duplicates += 1
                continue

            seen_texts.append(fingerprint)
            entries.append(build_memory_entry(item, generated_at=index.generated_at))

    entries.sort(key=lambda item: (item.priority_score, item.generated_at), reverse=True)
    pack.entries = entries[:max_entries] if max_entries is not None else entries
    pack.retained_entries = len(pack.entries)
    return pack


def discover_index_paths(analysis_dirs: list[Path]) -> list[Path]:
    """Find knowledge_index.json files in analysis output directories."""
    paths: list[Path] = []
    for directory in analysis_dirs:
        if directory.is_file() and directory.name == "knowledge_index.json":
            paths.append(directory)
            continue
        candidate = directory / "knowledge_index.json"
        if candidate.exists():
            paths.append(candidate)
    return sorted(unique_paths(paths))


def load_knowledge_index(path: Path) -> KnowledgeIndex:
    return KnowledgeIndex.model_validate_json(path.read_text(encoding="utf-8"))


def build_memory_entry(
    item: KnowledgeIndexEntry,
    generated_at: datetime,
) -> AgentMemoryEntry:
    return AgentMemoryEntry(
        source_url=item.source_url,
        source_title=item.source_title,
        card_title=item.card_title,
        claim=item.one_sentence,
        agent_move=item.agent_builder_takeaway,
        topics=item.topics,
        priority_score=item.priority_score,
        evidence=item.evidence[:2],
        generated_at=generated_at,
    )


def is_duplicate_memory(
    fingerprint: str,
    seen_texts: list[str],
    threshold: float,
) -> bool:
    return any(jaccard_similarity(fingerprint, seen) >= threshold for seen in seen_texts)


def memory_fingerprint(item: KnowledgeIndexEntry) -> str:
    return normalize_memory_text(
        " ".join(
            [
                item.card_title,
                item.one_sentence,
                item.agent_builder_takeaway,
                " ".join(topic.value for topic in item.topics),
            ]
        )
    )


def normalize_memory_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_after(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if len(normalized) == 10:
        normalized = f"{normalized}T00:00:00+00:00"
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def write_memory_pack(
    pack: AgentMemoryPack,
    out_dir: Path,
    include_evidence: bool = False,
    uncompressed_pack: AgentMemoryPack | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if uncompressed_pack is not None:
        write_uncompressed_memory_pack(uncompressed_pack, out_dir)
    write_json(out_dir / "agent_memory_pack.json", pack.model_dump(mode="json"))
    (out_dir / "agent_memory_pack.md").write_text(
        render_memory_pack(pack, include_evidence=include_evidence),
        encoding="utf-8",
    )
    compact = build_compact_memory_pack(pack, budget="compact")
    ultra = build_compact_memory_pack(pack, budget="ultra_compact")
    write_json(out_dir / "agent_memory_pack.compact.json", compact.model_dump(mode="json"))
    write_json(out_dir / "agent_memory_pack.ultra_compact.json", ultra.model_dump(mode="json"))
    (out_dir / "agent_memory_pack.compact.md").write_text(
        render_compact_memory_pack(compact),
        encoding="utf-8",
    )
    (out_dir / "agent_memory_pack.ultra_compact.md").write_text(
        render_compact_memory_pack(ultra),
        encoding="utf-8",
    )
    (out_dir / "frontier_learning_report.md").write_text(
        render_human_learning_report(pack),
        encoding="utf-8",
    )


def write_uncompressed_memory_pack(pack: AgentMemoryPack, out_dir: Path) -> None:
    """Write the long-term memory layer without entry caps or duplicate pruning."""
    write_json(out_dir / "agent_memory_pack.uncompressed.json", pack.model_dump(mode="json"))
    (out_dir / "agent_memory_pack.uncompressed.md").write_text(
        render_memory_pack(
            pack,
            include_evidence=True,
            title="Agent Memory Pack - Uncompressed Long-Term Layer",
        ),
        encoding="utf-8",
    )


def render_memory_pack(
    pack: AgentMemoryPack,
    include_evidence: bool = False,
    title: str = "Agent Memory Pack",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated at: {pack.generated_at.isoformat()}",
        f"After: {pack.after.isoformat() if pack.after else '(none)'}",
        f"Input entries: {pack.total_input_entries}",
        f"Retained entries: {pack.retained_entries}",
        (
            "Dropped: "
            f"time={pack.dropped_by_time}, "
            f"priority={pack.dropped_by_priority}, "
            f"duplicates={pack.dropped_duplicates}"
        ),
        "",
        "## Entries",
        "",
    ]
    if not pack.entries:
        lines.append("No memory entries passed the filters.")
        return "\n".join(lines).strip() + "\n"

    for index, entry in enumerate(pack.entries, start=1):
        topics = ", ".join(topic.value for topic in entry.topics)
        lines.extend(
            [
                f"### {index}. {entry.card_title}",
                "",
                f"- Time: {entry.generated_at.date().isoformat()}",
                f"- Priority: {entry.priority_score:.3f}",
                f"- Topics: {topics}",
                f"- Claim: {entry.claim}",
                f"- Agent move: {entry.agent_move}",
                f"- Source: {entry.source_url}",
            ]
        )
        if include_evidence and entry.evidence:
            evidence = " | ".join(entry.evidence)
            lines.append(f"- Evidence: {evidence}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_compact_memory_pack(
    pack: AgentMemoryPack,
    budget: str = "compact",
) -> CompactMemoryPack:
    """Build a compact distribution layer for direct agent context injection."""
    max_patterns = 8 if budget == "compact" else 4
    max_rules = 5 if budget == "compact" else 3
    entries = pack.entries[:max_patterns]
    return CompactMemoryPack(
        source_pack="agent_memory_pack.json",
        budget=budget,
        core_rules=derive_core_rules(entries)[:max_rules],
        current_patterns=derive_current_patterns(entries, max_items=max_patterns),
        anti_patterns=derive_anti_patterns(entries)[:max_rules],
        retrieval_pointers=derive_retrieval_pointers(entries)[:max_rules],
        watchlist=derive_watchlist(pack.entries[max_patterns:])[:3],
        memory_operations=derive_memory_operations(pack),
    )


def render_compact_memory_pack(pack: CompactMemoryPack) -> str:
    lines = [
        "# Agent Development Compact Memory",
        "",
        f"Budget: {pack.budget}",
        f"Scope: {pack.scope}",
        f"Generated at: {pack.generated_at.isoformat()}",
        "",
        "## Core Rules",
        "",
        *render_list_or_empty(pack.core_rules),
        "",
        "## Current Patterns",
        "",
        *render_list_or_empty(pack.current_patterns),
        "",
        "## Anti-Patterns",
        "",
        *render_list_or_empty(pack.anti_patterns),
        "",
        "## Retrieval Pointers",
        "",
        *render_list_or_empty(pack.retrieval_pointers),
    ]
    if pack.watchlist:
        lines.extend(["", "## Watchlist", ""])
        lines.extend(f"- {item}" for item in pack.watchlist)
    if pack.memory_operations:
        lines.extend(["", "## Memory Operations", ""])
        lines.extend(f"- {item}" for item in pack.memory_operations)
    return "\n".join(lines).strip() + "\n"


def render_list_or_empty(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- No retained items."]


def derive_core_rules(entries: list[AgentMemoryEntry]) -> list[str]:
    rules: list[str] = []
    topic_values = {topic.value for entry in entries for topic in entry.topics}
    if "mcp" in topic_values or "tool_use" in topic_values:
        rules.append(
            "Treat tools, data, prompts, and workflows as explicit integration boundaries."
        )
    if "multi_agent" in topic_values:
        rules.append(
            "Define ownership, shared state, and handoff contracts before delegating work."
        )
    if "agent_hardening" in topic_values or "safety" in topic_values:
        rules.append("Treat reliability, permissions, and failure modes as design requirements.")
    if "context_engineering" in topic_values:
        rules.append("Manage context explicitly with retrieval, compaction, and evidence budgets.")
    if "evaluation" in topic_values or "workflow" in topic_values:
        rules.append("Add validation gates before long-running agent workflows continue.")
    if "memory" in topic_values or "retrieval" in topic_values or "rag" in topic_values:
        rules.append("Keep source-grounded evidence with memories that affect agent behavior.")
    rules.append("Prefer authoritative, current sources over repeated lower-quality explanations.")
    return unique_strings(rules)


def derive_current_patterns(
    entries: list[AgentMemoryEntry],
    max_items: int,
) -> list[str]:
    patterns: list[str] = []
    for entry in entries[:max_items]:
        topics = ", ".join(topic.value for topic in entry.topics[:3])
        patterns.append(
            f"{entry.card_title}: {entry.claim} Agent move: {entry.agent_move} "
            f"[topics: {topics}; source: {entry.source_url}]"
        )
    return patterns


def derive_anti_patterns(entries: list[AgentMemoryEntry]) -> list[str]:
    anti_patterns = [
        "Do not treat popular demos as reusable agent knowledge unless they teach architecture.",
        (
            "Do not keep near-duplicate memories when a newer official source explains "
            "the idea better."
        ),
    ]
    topic_values = {topic.value for entry in entries for topic in entry.topics}
    if "tool_use" in topic_values:
        anti_patterns.append(
            "Do not hide tool contracts in prompt prose when schemas can be explicit."
        )
    if "memory" in topic_values or "retrieval" in topic_values:
        anti_patterns.append("Do not store untraceable claims as agent memory.")
    if "agent_hardening" in topic_values:
        anti_patterns.append(
            "Do not ship autonomous loops without failure-mode and permission checks."
        )
    return anti_patterns


def derive_retrieval_pointers(entries: list[AgentMemoryEntry]) -> list[str]:
    pointers: list[str] = []
    topics = sorted({topic.value for entry in entries for topic in entry.topics})
    for topic in topics[:8]:
        pointers.append(
            f"For details, retrieve cards with topic:{topic} from retrieval_manifest.json."
        )
    return pointers


def derive_watchlist(entries: list[AgentMemoryEntry]) -> list[str]:
    return [
        f"{entry.card_title}: keep as retrieval-only unless it becomes central to future runs."
        for entry in entries
    ]


def derive_memory_operations(pack: AgentMemoryPack) -> list[str]:
    return [
        f"ADD_OR_UPDATE retained={pack.retained_entries}",
        f"DROP_DUPLICATES count={pack.dropped_duplicates}",
        f"DROP_BY_TIME count={pack.dropped_by_time}",
        f"DROP_BY_PRIORITY count={pack.dropped_by_priority}",
    ]


def unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def render_human_learning_report(pack: AgentMemoryPack) -> str:
    """Render an English learning report for humans, with source URLs."""
    grouped_entries = group_entries_for_learning(pack.entries)
    lines = [
        "# Frontier Agent Development Learning Report",
        "",
        "This report is for human study, not agent memory injection. It turns the "
        "harvested knowledge into a readable learning path with source URLs.",
        "",
        "## Run Summary",
        "",
        f"- Generated at: {pack.generated_at.isoformat()}",
        f"- Knowledge cutoff filter: {pack.after.isoformat() if pack.after else '(none)'}",
        f"- Input entries: {pack.total_input_entries}",
        f"- Retained entries: {pack.retained_entries}",
        (
            "- Dropped entries: "
            f"time={pack.dropped_by_time}, "
            f"priority={pack.dropped_by_priority}, "
            f"duplicates={pack.dropped_duplicates}"
        ),
        "",
        "## Big Picture",
        "",
        build_learning_overview(pack.entries),
        "",
        "## How To Read This",
        "",
        "1. Read the theme summaries first.",
        "2. Open the source URLs only for themes you want to inspect deeply.",
        "3. Use the practice questions to turn the ideas into design checks for your own agents.",
        "",
    ]
    if not pack.entries:
        lines.append("No high-signal entries passed the current filters.")
        return "\n".join(lines).strip() + "\n"

    for theme, entries in grouped_entries.items():
        lines.extend(
            [
                f"## {theme}",
                "",
                theme_summary(theme, entries),
                "",
            ]
        )
        for index, entry in enumerate(entries, start=1):
            topics = ", ".join(topic.value for topic in entry.topics)
            lines.extend(
                [
                    f"### {index}. {entry.card_title}",
                    "",
                    f"**What to learn:** {entry.claim}",
                    "",
                    f"**Why it matters:** {learning_explanation(entry.agent_move)}",
                    "",
                    f"**How to apply it:** {entry.agent_move}",
                    "",
                    f"**Source:** {entry.source_url}",
                    "",
                    f"**Topics:** {topics}",
                    "",
                ]
            )
            if entry.evidence:
                lines.append("**Source evidence:**")
                lines.extend(f"- {evidence}" for evidence in entry.evidence)
                lines.append("")
            lines.append(f"**Practice question:** {practice_question(theme)}")
            lines.append("")

    lines.extend(
        [
            "## Notes On Quality",
            "",
            "- Priority combines relevance, frontier signal, topic breadth, and evidence presence.",
            (
                "- Jaccard-based de-duplication is only a coarse lexical guard; "
                "it should not be treated as a full judgment of whether two "
                "technical approaches are genuinely the same."
            ),
            "- Prefer source inspection for entries that look important but thinly evidenced.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def group_entries_for_learning(
    entries: list[AgentMemoryEntry],
) -> dict[str, list[AgentMemoryEntry]]:
    groups: dict[str, list[AgentMemoryEntry]] = {}
    for entry in entries:
        theme = learning_theme(entry)
        groups.setdefault(theme, []).append(entry)
    return groups


def learning_theme(entry: AgentMemoryEntry) -> str:
    topic_values = {topic.value for topic in entry.topics}
    if "mcp" in topic_values:
        return "Tool And Data Integration"
    if "workflow" in topic_values or "evaluation" in topic_values:
        return "Workflow Design And Validation"
    if "multi_agent" in topic_values:
        return "Multi-Agent Coordination"
    if "agent_hardening" in topic_values or "safety" in topic_values:
        return "Agent Hardening And Safety"
    if "context_engineering" in topic_values:
        return "Context Engineering"
    if "memory" in topic_values or "retrieval" in topic_values or "rag" in topic_values:
        return "Memory And Retrieval"
    return "Agent Architecture"


def build_learning_overview(entries: list[AgentMemoryEntry]) -> str:
    if not entries:
        return "No retained entries are available yet."
    themes = list(group_entries_for_learning(entries).keys())
    theme_text = ", ".join(themes)
    return (
        "The current harvested material mainly teaches how to connect agents to tools/data, "
        "structure long-running work into explicit phases, and preserve validation points. "
        f"The active learning themes are: {theme_text}."
    )


def theme_summary(theme: str, entries: list[AgentMemoryEntry]) -> str:
    if theme == "Tool And Data Integration":
        return (
            "Focus on how agents reach external systems. The main lesson is to treat "
            "connectors as explicit integration boundaries rather than hidden prompt text."
        )
    if theme == "Workflow Design And Validation":
        return (
            "Focus on how agents move from intent to implementation. The main lesson is to "
            "make phases, outputs, validation gates, and handoff state explicit."
        )
    if theme == "Multi-Agent Coordination":
        return (
            "Focus on delegation and coordination. The main lesson is to define who owns "
            "which state, when handoffs happen, and how success is checked."
        )
    if theme == "Memory And Retrieval":
        return (
            "Focus on what agents remember and retrieve. The main lesson is to preserve "
            "source-grounded context instead of letting memory become untraceable prose."
        )
    if theme == "Agent Hardening And Safety":
        return (
            "Focus on reliability, permissions, and failure modes. The main lesson is to "
            "design agent autonomy with explicit constraints and inspection points."
        )
    if theme == "Context Engineering":
        return (
            "Focus on what enters, leaves, and persists across the context window. The main "
            "lesson is to manage retrieval, compaction, and evidence as first-class design."
        )
    return (
        "Focus on reusable architecture choices. The main lesson is to turn broad ideas "
        "into concrete boundaries, state, tools, and checks."
    )


def learning_explanation(agent_move: str) -> str:
    if "integration boundary" in agent_move:
        return (
            "External tools and data are failure-prone boundaries. Making them explicit "
            "helps an agent reason about permissions, schemas, errors, and provenance."
        )
    if "validation gates" in agent_move:
        return (
            "Long-running agents drift unless each phase has a visible output and a way "
            "to decide whether the next phase should start."
        )
    if "handoff" in agent_move:
        return (
            "Multi-agent systems fail silently when ownership and shared state are vague; "
            "handoff rules make coordination inspectable."
        )
    if "source links" in agent_move or "retrieval context" in agent_move:
        return (
            "Memory is only useful when later agents can inspect where a claim came from "
            "and why it was retrieved."
        )
    return "This idea is useful when it changes how the agent is designed, tested, or operated."


def practice_question(theme: str) -> str:
    questions = {
        "Tool And Data Integration": (
            "Which tools/data sources in your agent should become explicit connectors with schemas?"
        ),
        "Workflow Design And Validation": (
            "Where should your agent pause, validate output, or record state before continuing?"
        ),
        "Multi-Agent Coordination": (
            "What state must cross the handoff, and which agent is accountable after the handoff?"
        ),
        "Memory And Retrieval": (
            "What evidence should be stored with each memory so a future agent can trust it?"
        ),
        "Agent Hardening And Safety": (
            "Which failure modes, permissions, and stop conditions should be explicit?"
        ),
        "Context Engineering": (
            "What should be retrieved, compacted, or dropped before the agent continues?"
        ),
        "Agent Architecture": (
            "What boundary or check would make this idea concrete in your own agent?"
        ),
    }
    return questions.get(theme, questions["Agent Architecture"])


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output
