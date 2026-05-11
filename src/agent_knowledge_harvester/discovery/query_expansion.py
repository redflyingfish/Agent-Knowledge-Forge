from pathlib import Path

from agent_knowledge_harvester.schemas.analysis import KnowledgeTopic
from agent_knowledge_harvester.schemas.discovery import SearchQueryPlan, TopicExpansion
from agent_knowledge_harvester.utils.files import write_json

TOPIC_SEED_TERMS: dict[KnowledgeTopic, list[str]] = {
    KnowledgeTopic.AGENT_ARCHITECTURE: ["agent architecture", "agent runtime", "agent framework"],
    KnowledgeTopic.AGENT_HARDENING: ["agent hardening", "agent robustness", "agent failure modes"],
    KnowledgeTopic.CODING_AGENTS: [
        "coding agents",
        "software engineering agents",
        "AI coding agents",
    ],
    KnowledgeTopic.COMPUTER_USE: ["computer use agents", "browser agents", "GUI agents"],
    KnowledgeTopic.CONTEXT_ENGINEERING: [
        "context engineering",
        "context compaction",
        "agent context management",
    ],
    KnowledgeTopic.MCP: ["model context protocol", "MCP server", "MCP client"],
    KnowledgeTopic.MEMORY: ["agent memory", "long-term memory agents", "semantic memory agents"],
    KnowledgeTopic.RAG: ["agentic RAG", "agent retrieval", "retrieval augmented agents"],
    KnowledgeTopic.RETRIEVAL: ["agent retrieval", "hybrid retrieval agents", "retrieval tool use"],
    KnowledgeTopic.MULTI_AGENT: ["multi-agent systems", "agent handoff", "agent coordination"],
    KnowledgeTopic.OBSERVABILITY: ["agent observability", "LLM tracing", "agent telemetry"],
    KnowledgeTopic.PLANNING: ["agent planning", "task decomposition agents", "reflection agents"],
    KnowledgeTopic.PROTOCOLS: ["agent protocol", "agent tool protocol", "LLM application protocol"],
    KnowledgeTopic.TOOL_USE: ["tool calling agents", "function calling agents", "agent tools"],
    KnowledgeTopic.SAFETY: ["agent safety", "tool safety agents", "agent permissions"],
    KnowledgeTopic.STRUCTURED_OUTPUTS: [
        "structured outputs agents",
        "JSON schema tool calling",
        "typed agent outputs",
    ],
    KnowledgeTopic.EVALUATION: ["agent evaluation", "agent evals", "agent benchmarks"],
    KnowledgeTopic.WORKFLOW: ["agent workflows", "durable agents", "workflow orchestration agents"],
    KnowledgeTopic.DEPLOYMENT: [
        "production agents",
        "agent deployment",
        "agent runtime operations",
    ],
}

ADJACENCY: dict[KnowledgeTopic, list[KnowledgeTopic]] = {
    KnowledgeTopic.MEMORY: [
        KnowledgeTopic.CONTEXT_ENGINEERING,
        KnowledgeTopic.RAG,
        KnowledgeTopic.RETRIEVAL,
        KnowledgeTopic.EVALUATION,
    ],
    KnowledgeTopic.RAG: [
        KnowledgeTopic.MEMORY,
        KnowledgeTopic.RETRIEVAL,
        KnowledgeTopic.CONTEXT_ENGINEERING,
        KnowledgeTopic.EVALUATION,
    ],
    KnowledgeTopic.AGENT_HARDENING: [
        KnowledgeTopic.SAFETY,
        KnowledgeTopic.OBSERVABILITY,
        KnowledgeTopic.EVALUATION,
        KnowledgeTopic.DEPLOYMENT,
    ],
    KnowledgeTopic.CODING_AGENTS: [
        KnowledgeTopic.TOOL_USE,
        KnowledgeTopic.WORKFLOW,
        KnowledgeTopic.EVALUATION,
        KnowledgeTopic.AGENT_HARDENING,
    ],
    KnowledgeTopic.MCP: [
        KnowledgeTopic.PROTOCOLS,
        KnowledgeTopic.TOOL_USE,
        KnowledgeTopic.SAFETY,
        KnowledgeTopic.CONTEXT_ENGINEERING,
    ],
    KnowledgeTopic.MULTI_AGENT: [
        KnowledgeTopic.PLANNING,
        KnowledgeTopic.WORKFLOW,
        KnowledgeTopic.OBSERVABILITY,
        KnowledgeTopic.EVALUATION,
    ],
}

DEFAULT_TOPICS = [
    KnowledgeTopic.MCP,
    KnowledgeTopic.TOOL_USE,
    KnowledgeTopic.MEMORY,
    KnowledgeTopic.RAG,
    KnowledgeTopic.CONTEXT_ENGINEERING,
    KnowledgeTopic.AGENT_HARDENING,
    KnowledgeTopic.EVALUATION,
    KnowledgeTopic.OBSERVABILITY,
    KnowledgeTopic.CODING_AGENTS,
    KnowledgeTopic.MULTI_AGENT,
    KnowledgeTopic.WORKFLOW,
    KnowledgeTopic.SAFETY,
    KnowledgeTopic.DEPLOYMENT,
]

SOURCE_QUALIFIERS = {
    "authority": ["official docs", "specification", "SDK docs", "technical report"],
    "implementation": ["GitHub", "open source", "reference implementation", "example repo"],
    "risk": ["failure modes", "best practices", "security", "evaluation"],
}


def build_query_plan(
    topics: list[KnowledgeTopic] | None = None,
    year: int = 2026,
) -> SearchQueryPlan:
    """Expand known topics and add scout queries for unknown frontier concepts."""
    selected_topics = topics or DEFAULT_TOPICS
    expansions = [build_topic_expansion(topic, year=year) for topic in selected_topics]
    return SearchQueryPlan(
        topic_expansions=expansions,
        frontier_scout_queries=build_frontier_scout_queries(year),
        source_hub_queries=build_source_hub_queries(year),
        stop_signal_queries=build_stop_signal_queries(year),
    )


def build_topic_expansion(topic: KnowledgeTopic, year: int) -> TopicExpansion:
    seed_terms = TOPIC_SEED_TERMS.get(topic, [topic.value.replace("_", " ")])
    adjacent_terms = []
    for adjacent in ADJACENCY.get(topic, []):
        adjacent_terms.extend(TOPIC_SEED_TERMS.get(adjacent, [adjacent.value.replace("_", " ")]))
    authority_queries = [
        f"{term} {qualifier} {year}"
        for term in seed_terms
        for qualifier in SOURCE_QUALIFIERS["authority"]
    ]
    implementation_queries = [
        f"{term} {qualifier} {year}"
        for term in seed_terms
        for qualifier in SOURCE_QUALIFIERS["implementation"]
    ]
    risk_queries = [
        f"{term} {qualifier} {year}"
        for term in seed_terms
        for qualifier in SOURCE_QUALIFIERS["risk"]
    ]
    return TopicExpansion(
        topic=topic.value,
        seed_terms=seed_terms,
        adjacent_terms=unique_strings(adjacent_terms),
        authority_queries=authority_queries,
        implementation_queries=implementation_queries,
        risk_queries=risk_queries,
    )


def build_frontier_scout_queries(year: int) -> list[str]:
    """Broad queries intentionally not tied to the fixed topic list."""
    return [
        f"frontier agent engineering {year} new pattern",
        f"AI agent systems {year} emerging architecture",
        f"agent development best practices {year} technical report",
        f"LLM agents {year} production lessons",
        f"agent framework release notes {year}",
        f"agent protocol benchmark memory tool use {year}",
    ]


def build_source_hub_queries(year: int) -> list[str]:
    return [
        f"site:docs.anthropic.com agents {year}",
        f"site:platform.openai.com agents tools memory {year}",
        f"site:langchain.com agent memory evaluation {year}",
        f"site:modelcontextprotocol.io specification {year}",
        f"site:github.com agent framework memory eval {year}",
        f"site:arxiv.org agent memory evaluation tool use {year}",
    ]


def build_stop_signal_queries(year: int) -> list[str]:
    return [
        f"agent development survey {year}",
        f"AI agents benchmark leaderboard {year}",
        f"agent systems production retrospective {year}",
    ]


def render_query_plan(plan: SearchQueryPlan) -> str:
    lines = [
        "# Frontier Agent Discovery Query Plan",
        "",
        f"Recency policy: {plan.recency_policy}",
        "",
        "## Topic Expansions",
        "",
    ]
    for expansion in plan.topic_expansions:
        lines.extend(
            [
                f"### {expansion.topic}",
                "",
                "Seed terms: " + ", ".join(expansion.seed_terms),
                "Adjacent terms: " + ", ".join(expansion.adjacent_terms or ["(none)"]),
                "",
                "Authority queries:",
                *[f"- {query}" for query in expansion.authority_queries],
                "",
                "Implementation queries:",
                *[f"- {query}" for query in expansion.implementation_queries],
                "",
                "Risk/evaluation queries:",
                *[f"- {query}" for query in expansion.risk_queries],
                "",
            ]
        )
    lines.extend(["## Frontier Scout Queries", ""])
    lines.extend(f"- {query}" for query in plan.frontier_scout_queries)
    lines.extend(["", "## Source Hub Queries", ""])
    lines.extend(f"- {query}" for query in plan.source_hub_queries)
    lines.extend(["", "## Stop-Signal Queries", ""])
    lines.extend(f"- {query}" for query in plan.stop_signal_queries)
    return "\n".join(lines).strip() + "\n"


def write_query_plan(plan: SearchQueryPlan, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "query_plan.json", plan.model_dump(mode="json"))
    (out_dir / "query_plan.md").write_text(render_query_plan(plan), encoding="utf-8")


def parse_topic_values(values: list[str] | None) -> list[KnowledgeTopic] | None:
    if not values:
        return None
    topics: list[KnowledgeTopic] = []
    for value in values:
        try:
            topics.append(KnowledgeTopic(value.strip()))
        except ValueError as exc:
            raise ValueError(f"unknown topic: {value}") from exc
    return topics


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
