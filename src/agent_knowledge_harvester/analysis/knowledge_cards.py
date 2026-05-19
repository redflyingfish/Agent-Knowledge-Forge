import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from agent_knowledge_harvester.schemas.analysis import (
    AnalysisResult,
    AnalysisRunStats,
    FrontierBrief,
    KnowledgeCard,
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)
from agent_knowledge_harvester.schemas.ingestion import CleanDocument, IngestionResult
from agent_knowledge_harvester.utils.files import stable_slug, write_json

TOPIC_KEYWORDS: dict[KnowledgeTopic, tuple[str, ...]] = {
    KnowledgeTopic.AGENT_ARCHITECTURE: (
        "agent",
        "assistant",
        "ai application",
        "autonomous",
        "planner",
        "executor",
    ),
    KnowledgeTopic.AGENT_HARDENING: (
        "hardening",
        "robustness",
        "failure mode",
        "guardrail",
        "reliability",
        "attack",
        "red team",
        "prompt injection",
    ),
    KnowledgeTopic.CODING_AGENTS: (
        "coding agent",
        "software engineering agent",
        "code agent",
        "program repair",
        "repository",
        "pull request",
        "codebase",
    ),
    KnowledgeTopic.COMPUTER_USE: (
        "computer use",
        "browser",
        "desktop",
        "gui",
        "screenshot",
        "click",
        "keyboard",
    ),
    KnowledgeTopic.CONTEXT_ENGINEERING: (
        "context engineering",
        "context window",
        "context management",
        "compaction",
        "summarization",
        "prompt caching",
    ),
    KnowledgeTopic.COST_LATENCY: (
        "cost",
        "latency",
        "token budget",
        "rate limit",
        "throughput",
        "batching",
        "caching",
    ),
    KnowledgeTopic.DATA_CONNECTORS: (
        "connector",
        "data source",
        "database",
        "calendar",
        "notion",
        "slack",
        "crm",
        "filesystem",
    ),
    KnowledgeTopic.DEPLOYMENT: (
        "deployment",
        "production",
        "runtime",
        "latency",
        "cost",
        "rate limit",
        "rollout",
    ),
    KnowledgeTopic.GUARDRAILS: (
        "guardrail",
        "policy",
        "moderation",
        "constraint",
        "validation",
        "approval",
        "allowlist",
        "denylist",
    ),
    KnowledgeTopic.HUMAN_IN_LOOP: (
        "human in the loop",
        "hitl",
        "human review",
        "approval",
        "interrupt",
        "escalation",
        "handover",
    ),
    KnowledgeTopic.IDENTITY_ACCESS: (
        "identity",
        "auth",
        "oauth",
        "permission",
        "access control",
        "credential",
        "secret",
        "tenant",
    ),
    KnowledgeTopic.KNOWLEDGE_GRAPHS: (
        "knowledge graph",
        "graph retrieval",
        "entity",
        "relationship",
        "ontology",
        "graph database",
    ),
    KnowledgeTopic.MCP: (
        "mcp",
        "model context protocol",
        "server",
        "client",
        "connector",
    ),
    KnowledgeTopic.MODEL_ROUTING: (
        "model routing",
        "router",
        "fallback model",
        "model selection",
        "small model",
        "large model",
        "mixture",
    ),
    KnowledgeTopic.MULTIMODAL_AGENTS: (
        "multimodal",
        "vision",
        "image",
        "screenshot",
        "audio",
        "video",
        "visual",
    ),
    KnowledgeTopic.MULTI_AGENT: (
        "multi-agent",
        "multi agent",
        "swarm",
        "handoff",
        "collaboration",
    ),
    KnowledgeTopic.OBSERVABILITY: (
        "observability",
        "trace",
        "tracing",
        "telemetry",
        "monitoring",
        "span",
        "debug",
    ),
    KnowledgeTopic.PLANNING: (
        "planning",
        "plan",
        "planner",
        "task decomposition",
        "reflection",
        "reasoning",
    ),
    KnowledgeTopic.PROMPT_ENGINEERING: (
        "prompt",
        "system prompt",
        "instruction",
        "few-shot",
        "prompt template",
        "prompt injection",
    ),
    KnowledgeTopic.PROTOCOLS: (
        "protocol",
        "specification",
        "schema",
        "interface",
        "contract",
        "standard",
    ),
    KnowledgeTopic.REASONING: (
        "reasoning",
        "chain of thought",
        "reflection",
        "self-critique",
        "deliberation",
        "tree of thought",
    ),
    KnowledgeTopic.STATE_RUNTIME: (
        "state",
        "runtime",
        "checkpoint",
        "session",
        "resume",
        "durable execution",
        "persistence",
    ),
    KnowledgeTopic.TOOL_USE: (
        "tool",
        "tools",
        "function calling",
        "tool calling",
        "workflow",
        "action",
    ),
    KnowledgeTopic.TOOL_ROUTING: (
        "tool routing",
        "tool selection",
        "tool choice",
        "router",
        "tool registry",
        "capability discovery",
    ),
    KnowledgeTopic.MEMORY: (
        "memory",
        "long-term memory",
        "episodic",
        "semantic memory",
        "procedural memory",
        "notion",
        "calendar",
        "database",
        "data source",
    ),
    KnowledgeTopic.RAG: (
        "rag",
        "retrieval-augmented",
        "retrieval augmented",
        "vector store",
        "embedding",
        "chunk",
    ),
    KnowledgeTopic.RETRIEVAL: (
        "retrieval",
        "rag",
        "search",
        "knowledge",
        "context",
    ),
    KnowledgeTopic.WORKFLOW: (
        "workflow",
        "orchestration",
        "durable",
        "integration",
        "automation",
    ),
    KnowledgeTopic.EVALUATION: (
        "evaluation",
        "eval",
        "evals",
        "benchmark",
        "test",
        "validation",
        "quality",
    ),
    KnowledgeTopic.SAFETY: (
        "safety",
        "permission",
        "consent",
        "sandbox",
        "policy",
        "data privacy",
        "tool safety",
    ),
    KnowledgeTopic.STRUCTURED_OUTPUTS: (
        "structured output",
        "json schema",
        "schema validation",
        "function schema",
        "tool schema",
        "typed output",
    ),
    KnowledgeTopic.FRONTIER_SIGNAL: (
        "open-source standard",
        "supported",
        "ecosystem",
        "claude",
        "chatgpt",
        "cursor",
        "visual studio code",
    ),
}

HIGH_VALUE_PATTERNS = (
    "is an open-source standard",
    "can connect",
    "provides access",
    "reduces development time",
    "build once",
    "integrate everywhere",
    "acting as",
    "take actions",
)

LOW_VALUE_SECTION_TERMS = (
    "navigation menu",
    "folders and files",
    "latest commit",
    "use saved searches",
    "prerequisite",
    "installation",
    "install",
    "setup",
    "verify in",
    "option ",
    "agents.md",
    ".claude directory",
    "kiro",
    "amazon q",
    "cursor ide",
    "cline",
    "claude code",
    "github copilot",
    "openai codex",
    "other agents",
    "troubleshooting",
    "general issues",
    "platform-specific issues",
    "additional resources",
    "resources",
    "contributing",
    "license",
    "about",
    "stars",
    "watchers",
    "forks",
    "releases",
    "contributors",
    "languages",
    "footer",
)

LOW_VALUE_SENTENCE_PATTERNS = (
    "installation link",
    "install.",
    "installed:",
    "download the latest release",
    "paste this prompt",
    "under project rules",
    "directory structure",
    "start claude",
    "ask claude",
    "replace downloads",
    "automatically detected",
    "copy-item",
    "get-content",
    "out-file",
    "curl -",
    "reload this page",
)

T = TypeVar("T")


class KnowledgeCardAnalyzer:
    """Convert cleaned documents into compact agent-engineering knowledge cards."""

    def __init__(self, min_relevance_score: float = 0.18, max_cards_per_doc: int = 4) -> None:
        self.min_relevance_score = min_relevance_score
        self.max_cards_per_doc = max_cards_per_doc

    def analyze_document(self, document: CleanDocument) -> AnalysisResult:
        sections = split_markdown_sections(document.markdown)
        candidates = [
            self._build_card(document, heading, body)
            for heading, body in sections
            if body.strip()
            and not is_low_value_section(heading)
        ]
        cards = [
            card
            for card in sorted(
                candidates,
                key=lambda item: (item.frontier_score, item.relevance_score),
                reverse=True,
            )
            if card.relevance_score >= self.min_relevance_score
        ][: self.max_cards_per_doc]

        skipped_reason = None if cards else "no section passed the relevance threshold"
        return AnalysisResult(
            source_url=document.source_url,
            title=document.title,
            cards=cards,
            skipped_reason=skipped_reason,
        )

    def analyze_ingestion_result(self, result: IngestionResult) -> AnalysisResult | None:
        if not result.success or result.clean is None:
            return None
        return self.analyze_document(result.clean)

    def _build_card(self, document: CleanDocument, heading: str, body: str) -> KnowledgeCard:
        sentences = split_sentences(body)
        topics = detect_topics(" ".join([heading, body]))
        relevance_score = score_relevance(body, topics) if sentences else 0.0
        frontier_score = score_frontier(body) if sentences else 0.0
        evidence = select_evidence(sentences)
        one_sentence = select_one_sentence(sentences, document.summary_hint)
        why_it_matters = select_why_it_matters(sentences, one_sentence)
        implementation_notes = select_implementation_notes(sentences)

        return KnowledgeCard(
            source_url=document.source_url,
            title=heading or document.title or str(document.source_url),
            one_sentence=one_sentence,
            why_it_matters=why_it_matters,
            agent_builder_takeaway=build_agent_builder_takeaway(topics, one_sentence),
            topics=topics,
            implementation_notes=implementation_notes,
            evidence=evidence,
            relevance_score=relevance_score,
            frontier_score=frontier_score,
        )


def analyze_directory(
    in_dir: Path,
    out_dir: Path,
    analyzer: KnowledgeCardAnalyzer | None = None,
    max_index_entries: int | None = None,
) -> tuple[list[AnalysisResult], AnalysisRunStats, KnowledgeIndex]:
    analyzer = analyzer or KnowledgeCardAnalyzer()
    results: list[AnalysisResult] = []

    for path in iter_ingestion_json_files(in_dir):
        result = load_ingestion_result(path)
        if result is None:
            continue
        analysis = analyzer.analyze_ingestion_result(result)
        if analysis is not None:
            results.append(analysis)

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
    (out_dir / "knowledge_index.rich.md").write_text(
        render_rich_index_markdown(index),
        encoding="utf-8",
    )
    brief = build_frontier_brief(index)
    write_json(out_dir / "frontier_brief.json", brief.model_dump(mode="json"))
    (out_dir / "frontier_brief.md").write_text(render_frontier_brief(brief), encoding="utf-8")
    return results, stats, index


def load_ingestion_result(path: Path) -> IngestionResult | None:
    try:
        return IngestionResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None


def write_frontier_brief_from_index(index_path: Path, out_dir: Path | None = None) -> FrontierBrief:
    index = KnowledgeIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    brief = build_frontier_brief(index)
    target_dir = out_dir or index_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "frontier_brief.json", brief.model_dump(mode="json"))
    (target_dir / "frontier_brief.md").write_text(render_frontier_brief(brief), encoding="utf-8")
    return brief


def build_knowledge_index(
    results: list[AnalysisResult],
    max_entries: int | None = None,
) -> KnowledgeIndex:
    entries: list[KnowledgeIndexEntry] = []
    topic_counts: dict[KnowledgeTopic, int] = {}

    for result in results:
        for card in result.cards:
            for topic in card.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            entries.append(
                KnowledgeIndexEntry(
                    source_url=result.source_url,
                    source_title=result.title,
                    card_title=card.title,
                    one_sentence=card.one_sentence,
                    why_it_matters=card.why_it_matters,
                    agent_builder_takeaway=card.agent_builder_takeaway,
                    topics=card.topics,
                    implementation_notes=card.implementation_notes[:4],
                    relevance_score=card.relevance_score,
                    frontier_score=card.frontier_score,
                    priority_score=score_priority(card),
                    evidence=card.evidence[:2],
                )
            )

    entries.sort(key=lambda item: item.priority_score, reverse=True)
    return KnowledgeIndex(
        total_documents=len(results),
        total_cards=sum(len(result.cards) for result in results),
        topic_counts=dict(
            sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        entries=entries if max_entries is None else entries[:max_entries],
    )


def score_priority(card: KnowledgeCard) -> float:
    topic_bonus = min(len(card.topics) * 0.015, 0.09)
    evidence_bonus = 0.03 if card.evidence else 0.0
    score = (
        (card.relevance_score * 0.58)
        + (card.frontier_score * 0.34)
        + topic_bonus
        + evidence_bonus
    )
    return round(min(1.0, score), 3)


def build_frontier_brief(index: KnowledgeIndex, max_items: int = 5) -> FrontierBrief:
    top_entries = index.entries[:max_items]
    top_signals = unique_preserving_order(
        entry.one_sentence for entry in top_entries if entry.one_sentence
    )[:max_items]
    agent_builder_moves = unique_preserving_order(
        entry.agent_builder_takeaway for entry in top_entries if entry.agent_builder_takeaway
    )[:max_items]
    watch_topics = [topic for topic, _ in list(index.topic_counts.items())[:5]]
    source_urls = unique_preserving_order(entry.source_url for entry in top_entries)

    summary = build_brief_summary(top_entries, watch_topics)
    return FrontierBrief(
        summary=summary,
        top_signals=top_signals,
        agent_builder_moves=agent_builder_moves,
        next_experiments=suggest_next_experiments(top_entries),
        watch_topics=watch_topics,
        source_urls=source_urls,
    )


def build_brief_summary(
    entries: list[KnowledgeIndexEntry],
    watch_topics: list[KnowledgeTopic],
) -> str:
    if not entries:
        return "No high-signal frontier agent knowledge was found in this batch."

    leading = entries[0]
    topic_text = ", ".join(topic.value for topic in leading.topics[:3])
    if not topic_text:
        topic_text = ", ".join(topic.value for topic in watch_topics[:3]) or "agent engineering"
    return (
        f"Top priority is '{leading.card_title}', centered on {topic_text}. "
        f"For an Agent builder, the immediate move is: {leading.agent_builder_takeaway}"
    )


def suggest_next_experiments(entries: list[KnowledgeIndexEntry], max_items: int = 4) -> list[str]:
    suggestions: list[str] = []
    all_topics = {topic for entry in entries for topic in entry.topics}

    if KnowledgeTopic.MCP in all_topics:
        suggestions.append("Map one target workflow into data sources, tools, and MCP connectors.")
    if KnowledgeTopic.MULTI_AGENT in all_topics:
        suggestions.append(
            "Prototype the smallest handoff loop and record what state crosses agents."
        )
    if KnowledgeTopic.TOOL_USE in all_topics:
        suggestions.append(
            "Convert one repeated action into a typed tool with clear failure handling."
        )
    if KnowledgeTopic.MEMORY in all_topics or KnowledgeTopic.RETRIEVAL in all_topics:
        suggestions.append(
            "Test whether retrieved context preserves enough source evidence for action."
        )
    if KnowledgeTopic.EVALUATION in all_topics:
        suggestions.append("Add a lightweight evaluation case before scaling the workflow.")

    if not suggestions and entries:
        suggestions.append("Turn the top card into one small implementation spike.")

    return suggestions[:max_items]


def unique_preserving_order(items: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    output: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def iter_ingestion_json_files(in_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in in_dir.glob("*.json")
        if path.name not in {"run_stats.json", "analysis_stats.json"}
        and not path.name.endswith(".knowledge.json")
    )


def split_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    lines = markdown.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        heading = parse_heading(line)
        if heading is not None:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = heading
            current_body = []
            continue
        current_body.append(line)

    if current_body:
        sections.append((current_heading, current_body))

    if not sections and markdown.strip():
        return [("", markdown.strip())]

    return [
        (heading, "\n".join(body).strip())
        for heading, body in sections
        if "\n".join(body).strip()
    ]


def parse_heading(line: str) -> str | None:
    match = re.match(r"^#{1,4}\s+(.+)$", line.strip())
    if not match:
        return None
    return strip_markdown(match.group(1))


def is_low_value_section(heading: str) -> bool:
    lowered = heading.lower()
    if not lowered:
        return False
    return any(term in lowered for term in LOW_VALUE_SECTION_TERMS)


def split_sentences(text: str) -> list[str]:
    flattened = re.sub(r"\s+", " ", strip_markdown(normalize_table_rows(text))).strip()
    flattened = re.sub(r"\.{2,}", ".", flattened)
    flattened = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", flattened)
    flattened = flattened.replace("e.g.", "e<dot>g<dot>").replace("i.e.", "i<dot>e<dot>")
    if not flattened:
        return []
    candidates = re.split(r"(?<=[.!?])\s+", flattened)
    return [
        candidate.replace("<dot>", ".").strip()
        for candidate in candidates
        if len(candidate.replace("<dot>", ".").strip()) >= 20
        and not is_low_value_sentence(candidate)
        and not looks_like_file_listing(candidate.replace("<dot>", ".").strip())
    ]


def normalize_table_rows(text: str) -> str:
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            rows.append(line)
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|") if cell.strip()]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) >= 2 and len(cells) % 2 == 0:
            pairs = [
                f"{cells[index]}: {cells[index + 1]}"
                for index in range(0, len(cells), 2)
            ]
            rows.append(". ".join(pairs).rstrip(".") + ".")
            continue
        rows.append(". ".join(cells).rstrip(".") + ".")
    return "\n".join(rows)


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def is_low_value_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(
        noise in lowered
        for noise in (
            "navigation menu",
            "toggle navigation",
            "appearance settings",
            "github copilot write better code",
            "latest commit",
            "last commit",
            "commits",
            "sign in",
            "sign up",
            *LOW_VALUE_SENTENCE_PATTERNS,
        )
    )


def detect_topics(text: str) -> list[KnowledgeTopic]:
    lowered = text.lower()
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return topics or [KnowledgeTopic.AGENT_ARCHITECTURE]


def score_relevance(text: str, topics: list[KnowledgeTopic]) -> float:
    lowered = text.lower()
    topic_hits = sum(
        1
        for topic in topics
        for keyword in TOPIC_KEYWORDS[topic]
        if keyword in lowered
    )
    high_value_hits = sum(1 for pattern in HIGH_VALUE_PATTERNS if pattern in lowered)
    code_blocks = lowered.count("```") // 2
    score = (topic_hits * 0.055) + (high_value_hits * 0.09) + (code_blocks * 0.08)
    return round(min(1.0, score), 3)


def score_frontier(text: str) -> float:
    lowered = text.lower()
    support_signals = sum(
        1
        for keyword in TOPIC_KEYWORDS[KnowledgeTopic.FRONTIER_SIGNAL]
        if keyword in lowered
    )
    standard_signal = 1 if "standard" in lowered or "protocol" in lowered else 0
    ecosystem_signal = 1 if "ecosystem" in lowered or "supported" in lowered else 0
    score = (support_signals * 0.09) + (standard_signal * 0.15) + (ecosystem_signal * 0.14)
    return round(min(1.0, score), 3)


def select_one_sentence(sentences: list[str], fallback: str) -> str:
    if not sentences:
        return clamp_text(strip_markdown(normalize_table_rows(fallback)), 240)
    return clamp_text(max(sentences, key=sentence_value), 260)


def select_why_it_matters(sentences: list[str], one_sentence: str) -> str:
    for sentence in sentences:
        lowered = sentence.lower()
        if sentence != one_sentence and any(
            signal in lowered for signal in ("reduces", "enables", "provides", "capable", "access")
        ):
            return clamp_text(sentence, 260)
    return clamp_text(one_sentence, 260)


def select_implementation_notes(sentences: list[str], limit: int = 4) -> list[str]:
    notes: list[str] = []
    implementation_signals = (
        "connect",
        "build",
        "integrat",
        "access",
        "tool",
        "route",
        "cache",
        "store",
        "trace",
        "monitor",
        "evaluate",
        "test",
        "checkpoint",
        "schema",
        "permission",
        "auth",
        "guardrail",
        "sandbox",
        "memory",
        "retrieval",
        "orchestrat",
        "workflow",
    )
    for sentence in sentences:
        lowered = sentence.lower()
        if any(signal in lowered for signal in implementation_signals):
            notes.append(clamp_text(sentence, 280))
        if len(notes) >= limit:
            break
    return notes


def select_evidence(sentences: list[str], limit: int = 3) -> list[str]:
    ranked = sorted(sentences, key=sentence_value, reverse=True)
    return [clamp_text(sentence, 220) for sentence in ranked[:limit]]


def sentence_value(sentence: str) -> float:
    lowered = sentence.lower()
    if looks_like_file_listing(sentence):
        return 0.0
    topic_hits = sum(
        1
        for keywords in TOPIC_KEYWORDS.values()
        for keyword in keywords
        if keyword in lowered
    )
    high_value_hits = sum(1 for pattern in HIGH_VALUE_PATTERNS if pattern in lowered)
    length_bonus = min(len(sentence) / 240, 1.0)
    return topic_hits + (high_value_hits * 1.5) + length_bonus


def looks_like_file_listing(sentence: str) -> bool:
    lowered = sentence.lower()
    path_markers = lowered.count("/") + lowered.count("<slug>") + lowered.count(".github")
    if path_markers >= 2:
        return True
    return bool(re.fullmatch(r"[\w./-]+:\s*[\w./-]+\.?", sentence.strip()))


def build_agent_builder_takeaway(topics: list[KnowledgeTopic], one_sentence: str) -> str:
    if KnowledgeTopic.MCP in topics:
        return "Treat MCP as an integration boundary for tools, data, and workflows."
    if KnowledgeTopic.MULTI_AGENT in topics:
        return "Look for explicit coordination, handoff, and shared-state design choices."
    if KnowledgeTopic.WORKFLOW in topics:
        return (
            "Model the work as explicit phases with outputs, validation gates, "
            "and handoff state."
        )
    if KnowledgeTopic.TOOL_USE in topics:
        return "Convert the idea into callable tools with clear inputs, outputs, and failure modes."
    if KnowledgeTopic.MEMORY in topics or KnowledgeTopic.RETRIEVAL in topics:
        return "Preserve source links and retrieval context so agents can ground later actions."
    return clamp_text(one_sentence, 220)


def clamp_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rsplit(" ", 1)[0].strip(" ,;:-")
    if len(clipped) < max_chars * 0.6:
        clipped = text[:max_chars].strip(" ,;:-")
    return clipped + "..."


def render_index_markdown(index: KnowledgeIndex) -> str:
    lines = [
        "# Knowledge Index",
        "",
        f"Total documents: {index.total_documents}",
        f"Total cards: {index.total_cards}",
        "",
    ]

    if index.topic_counts:
        lines.append("## Topic Distribution")
        lines.append("")
        for topic, count in index.topic_counts.items():
            lines.append(f"- {topic.value}: {count}")
        lines.append("")

    lines.append("## Priority Cards")
    lines.append("")
    for position, entry in enumerate(index.entries, start=1):
        lines.extend(
            [
                f"### {position}. {entry.card_title}",
                "",
                f"Source: {entry.source_url}",
                f"Priority: {entry.priority_score:.3f} "
                f"(relevance={entry.relevance_score:.3f}, frontier={entry.frontier_score:.3f})",
                f"Topics: {', '.join(topic.value for topic in entry.topics)}",
                "",
                f"One sentence: {entry.one_sentence}",
                "",
                f"Why it matters: {entry.why_it_matters or entry.one_sentence}",
                "",
                f"Agent builder takeaway: {entry.agent_builder_takeaway}",
                "",
            ]
        )
        if entry.implementation_notes:
            lines.append("Implementation details:")
            lines.extend(f"- {note}" for note in entry.implementation_notes)
            lines.append("")
        if entry.evidence:
            lines.append("Evidence:")
            lines.extend(f"- {quote}" for quote in entry.evidence)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_rich_index_markdown(index: KnowledgeIndex, max_topic_cards: int = 8) -> str:
    lines = [
        "# Knowledge Index Rich View",
        "",
        f"Total documents: {index.total_documents}",
        f"Total cards: {index.total_cards}",
        f"Unique sources: {len({str(entry.source_url) for entry in index.entries})}",
        "",
    ]

    if index.topic_counts:
        lines.append("## Topic Coverage")
        lines.append("")
        for topic, count in index.topic_counts.items():
            lines.append(f"- {topic.value}: {count}")
        lines.append("")

    source_counts: dict[str, int] = {}
    source_titles: dict[str, str] = {}
    for entry in index.entries:
        source = str(entry.source_url)
        source_counts[source] = source_counts.get(source, 0) + 1
        if entry.source_title:
            source_titles[source] = entry.source_title

    if source_counts:
        lines.append("## Sources")
        lines.append("")
        for source, count in sorted(
            source_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            lines.append(f"- {source_titles.get(source, source)} ({count} cards)")
            lines.append(f"  - {source}")
        lines.append("")

    lines.append("## Priority Cards")
    lines.append("")
    for position, entry in enumerate(index.entries, start=1):
        lines.extend(
            [
                f"### {position}. {entry.card_title}",
                "",
                f"- Source: {entry.source_title or entry.source_url}",
                f"- URL: {entry.source_url}",
                (
                    "- Scores: "
                    f"priority={entry.priority_score:.3f}, "
                    f"relevance={entry.relevance_score:.3f}, "
                    f"frontier={entry.frontier_score:.3f}"
                ),
                f"- Topics: {', '.join(topic.value for topic in entry.topics) or '(none)'}",
                "",
                f"One sentence: {entry.one_sentence}",
                "",
                f"Why it matters: {entry.why_it_matters or entry.one_sentence}",
                "",
                f"Agent builder takeaway: {entry.agent_builder_takeaway}",
                "",
            ]
        )
        if entry.implementation_notes:
            lines.append("Implementation details:")
            lines.extend(f"- {note}" for note in entry.implementation_notes)
            lines.append("")
        if entry.evidence:
            lines.append("Evidence:")
            lines.extend(f"- {quote}" for quote in entry.evidence)
            lines.append("")

    if index.topic_counts:
        lines.append("## Topic Spotlight")
        lines.append("")
        for topic, _count in list(index.topic_counts.items())[:10]:
            topic_entries = [
                entry for entry in index.entries if topic in entry.topics
            ][:max_topic_cards]
            if not topic_entries:
                continue
            lines.extend([f"### {topic.value}", ""])
            for entry in topic_entries:
                lines.extend(
                    [
                        f"- {entry.card_title}",
                        f"  - {entry.agent_builder_takeaway}",
                        f"  - {entry.source_title or entry.source_url}",
                    ]
                )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_frontier_brief(brief: FrontierBrief) -> str:
    lines = [
        f"# {brief.title}",
        "",
        brief.summary,
        "",
    ]

    if brief.top_signals:
        lines.append("## Top Signals")
        lines.append("")
        lines.extend(f"- {signal}" for signal in brief.top_signals)
        lines.append("")

    if brief.agent_builder_moves:
        lines.append("## Agent Builder Moves")
        lines.append("")
        lines.extend(f"- {move}" for move in brief.agent_builder_moves)
        lines.append("")

    if brief.next_experiments:
        lines.append("## Next Experiments")
        lines.append("")
        lines.extend(f"- {experiment}" for experiment in brief.next_experiments)
        lines.append("")

    if brief.watch_topics:
        lines.append("## Watch Topics")
        lines.append("")
        lines.extend(f"- {topic.value}" for topic in brief.watch_topics)
        lines.append("")

    if brief.source_urls:
        lines.append("## Sources")
        lines.append("")
        lines.extend(f"- {url}" for url in brief.source_urls)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_markdown(result: AnalysisResult) -> str:
    title = result.title or str(result.source_url)
    lines = [f"# {title}", "", f"Source: {result.source_url}", ""]
    if result.skipped_reason:
        lines.extend([f"Skipped: {result.skipped_reason}", ""])

    for index, card in enumerate(result.cards, start=1):
        lines.extend(
            [
                f"## {index}. {card.title}",
                "",
                f"One sentence: {card.one_sentence}",
                "",
                f"Why it matters: {card.why_it_matters}",
                "",
                f"Agent builder takeaway: {card.agent_builder_takeaway}",
                "",
                "Topics: " + ", ".join(topic.value for topic in card.topics),
                f"Scores: relevance={card.relevance_score:.3f}, frontier={card.frontier_score:.3f}",
                "",
            ]
        )
        if card.implementation_notes:
            lines.append("Implementation notes:")
            lines.extend(f"- {note}" for note in card.implementation_notes)
            lines.append("")
        if card.evidence:
            lines.append("Evidence:")
            lines.extend(f"- {quote}" for quote in card.evidence)
            lines.append("")

    return "\n".join(lines).strip() + "\n"
