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
    KnowledgeTopic.COST_LATENCY: [
        "agent cost optimization",
        "agent latency optimization",
        "LLM agent throughput",
    ],
    KnowledgeTopic.DATA_CONNECTORS: [
        "agent data connectors",
        "agent database connectors",
        "agent enterprise integrations",
    ],
    KnowledgeTopic.MCP: ["model context protocol", "MCP server", "MCP client"],
    KnowledgeTopic.GUARDRAILS: [
        "agent guardrails",
        "AI agent policy enforcement",
        "agent constraints",
    ],
    KnowledgeTopic.HUMAN_IN_LOOP: [
        "human in the loop agents",
        "agent approval workflow",
        "agent human review",
    ],
    KnowledgeTopic.IDENTITY_ACCESS: [
        "agent identity access control",
        "agent OAuth permissions",
        "AI agent secrets management",
    ],
    KnowledgeTopic.KNOWLEDGE_GRAPHS: [
        "knowledge graph agents",
        "graph RAG agents",
        "entity memory agents",
    ],
    KnowledgeTopic.MEMORY: ["agent memory", "long-term memory agents", "semantic memory agents"],
    KnowledgeTopic.MODEL_ROUTING: [
        "LLM model routing agents",
        "agent model fallback",
        "multi model agent routing",
    ],
    KnowledgeTopic.MULTIMODAL_AGENTS: [
        "multimodal agents",
        "vision language agents",
        "screen understanding agents",
    ],
    KnowledgeTopic.RAG: ["agentic RAG", "agent retrieval", "retrieval augmented agents"],
    KnowledgeTopic.RETRIEVAL: ["agent retrieval", "hybrid retrieval agents", "retrieval tool use"],
    KnowledgeTopic.MULTI_AGENT: ["multi-agent systems", "agent handoff", "agent coordination"],
    KnowledgeTopic.OBSERVABILITY: ["agent observability", "LLM tracing", "agent telemetry"],
    KnowledgeTopic.PLANNING: ["agent planning", "task decomposition agents", "reflection agents"],
    KnowledgeTopic.PROMPT_ENGINEERING: [
        "agent system prompts",
        "agent prompt engineering",
        "prompt injection agents",
    ],
    KnowledgeTopic.PROTOCOLS: ["agent protocol", "agent tool protocol", "LLM application protocol"],
    KnowledgeTopic.REASONING: [
        "agent reasoning",
        "reflective agents",
        "deliberative agents",
    ],
    KnowledgeTopic.STATE_RUNTIME: [
        "stateful agent runtime",
        "agent checkpointing",
        "durable execution agents",
    ],
    KnowledgeTopic.TOOL_USE: ["tool calling agents", "function calling agents", "agent tools"],
    KnowledgeTopic.TOOL_ROUTING: [
        "agent tool routing",
        "agent tool selection",
        "tool registry agents",
    ],
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

CHINESE_TOPIC_TERMS: dict[KnowledgeTopic, list[str]] = {
    KnowledgeTopic.AGENT_ARCHITECTURE: ["AI Agent 架构", "大模型智能体架构"],
    KnowledgeTopic.CODING_AGENTS: ["编程智能体", "代码智能体", "AI 编程助手"],
    KnowledgeTopic.COMPUTER_USE: ["电脑使用智能体", "浏览器智能体", "GUI 智能体"],
    KnowledgeTopic.CONTEXT_ENGINEERING: ["上下文工程", "上下文压缩", "智能体上下文管理"],
    KnowledgeTopic.MCP: ["模型上下文协议", "MCP 服务器", "MCP 客户端"],
    KnowledgeTopic.GUARDRAILS: ["智能体护栏", "大模型安全护栏"],
    KnowledgeTopic.HUMAN_IN_LOOP: ["人在回路智能体", "人工审批智能体"],
    KnowledgeTopic.MEMORY: ["智能体记忆", "长期记忆智能体", "大模型记忆"],
    KnowledgeTopic.RAG: ["智能体 RAG", "Agentic RAG", "检索增强智能体"],
    KnowledgeTopic.RETRIEVAL: ["智能体检索", "混合检索智能体"],
    KnowledgeTopic.MULTI_AGENT: ["多智能体系统", "智能体协作", "智能体交接"],
    KnowledgeTopic.OBSERVABILITY: ["智能体可观测性", "LLM tracing", "智能体日志追踪"],
    KnowledgeTopic.PLANNING: ["智能体规划", "任务分解智能体", "反思智能体"],
    KnowledgeTopic.PROMPT_ENGINEERING: ["智能体提示词工程", "系统提示词", "提示注入智能体"],
    KnowledgeTopic.REASONING: ["智能体推理", "反思智能体", "规划推理智能体"],
    KnowledgeTopic.STATE_RUNTIME: ["有状态智能体运行时", "智能体检查点", "持久执行智能体"],
    KnowledgeTopic.TOOL_USE: ["工具调用智能体", "函数调用智能体", "智能体工具"],
    KnowledgeTopic.TOOL_ROUTING: ["智能体工具路由", "工具选择智能体"],
    KnowledgeTopic.SAFETY: ["智能体安全", "工具调用安全", "智能体权限"],
    KnowledgeTopic.STRUCTURED_OUTPUTS: ["结构化输出智能体", "JSON schema 工具调用"],
    KnowledgeTopic.EVALUATION: ["智能体评估", "智能体 benchmark", "智能体测试"],
    KnowledgeTopic.WORKFLOW: ["智能体工作流", "持久智能体", "工作流编排智能体"],
    KnowledgeTopic.DEPLOYMENT: ["生产级智能体", "智能体部署", "智能体运维"],
}

ADJACENCY: dict[KnowledgeTopic, list[KnowledgeTopic]] = {
    KnowledgeTopic.MEMORY: [
        KnowledgeTopic.CONTEXT_ENGINEERING,
        KnowledgeTopic.RAG,
        KnowledgeTopic.RETRIEVAL,
        KnowledgeTopic.KNOWLEDGE_GRAPHS,
        KnowledgeTopic.EVALUATION,
    ],
    KnowledgeTopic.RAG: [
        KnowledgeTopic.MEMORY,
        KnowledgeTopic.RETRIEVAL,
        KnowledgeTopic.CONTEXT_ENGINEERING,
        KnowledgeTopic.KNOWLEDGE_GRAPHS,
        KnowledgeTopic.EVALUATION,
    ],
    KnowledgeTopic.AGENT_HARDENING: [
        KnowledgeTopic.SAFETY,
        KnowledgeTopic.GUARDRAILS,
        KnowledgeTopic.OBSERVABILITY,
        KnowledgeTopic.EVALUATION,
        KnowledgeTopic.DEPLOYMENT,
        KnowledgeTopic.IDENTITY_ACCESS,
    ],
    KnowledgeTopic.CODING_AGENTS: [
        KnowledgeTopic.TOOL_USE,
        KnowledgeTopic.STATE_RUNTIME,
        KnowledgeTopic.WORKFLOW,
        KnowledgeTopic.EVALUATION,
        KnowledgeTopic.AGENT_HARDENING,
    ],
    KnowledgeTopic.MCP: [
        KnowledgeTopic.PROTOCOLS,
        KnowledgeTopic.TOOL_USE,
        KnowledgeTopic.SAFETY,
        KnowledgeTopic.DATA_CONNECTORS,
        KnowledgeTopic.CONTEXT_ENGINEERING,
    ],
    KnowledgeTopic.MULTI_AGENT: [
        KnowledgeTopic.PLANNING,
        KnowledgeTopic.REASONING,
        KnowledgeTopic.WORKFLOW,
        KnowledgeTopic.OBSERVABILITY,
        KnowledgeTopic.EVALUATION,
    ],
    KnowledgeTopic.TOOL_USE: [
        KnowledgeTopic.TOOL_ROUTING,
        KnowledgeTopic.STRUCTURED_OUTPUTS,
        KnowledgeTopic.GUARDRAILS,
        KnowledgeTopic.IDENTITY_ACCESS,
    ],
    KnowledgeTopic.DEPLOYMENT: [
        KnowledgeTopic.STATE_RUNTIME,
        KnowledgeTopic.COST_LATENCY,
        KnowledgeTopic.OBSERVABILITY,
        KnowledgeTopic.AGENT_HARDENING,
    ],
    KnowledgeTopic.COMPUTER_USE: [
        KnowledgeTopic.MULTIMODAL_AGENTS,
        KnowledgeTopic.TOOL_USE,
        KnowledgeTopic.SAFETY,
        KnowledgeTopic.OBSERVABILITY,
    ],
}

DEFAULT_TOPICS = [
    KnowledgeTopic.MCP,
    KnowledgeTopic.TOOL_USE,
    KnowledgeTopic.MEMORY,
    KnowledgeTopic.RAG,
    KnowledgeTopic.CONTEXT_ENGINEERING,
    KnowledgeTopic.STATE_RUNTIME,
    KnowledgeTopic.REASONING,
    KnowledgeTopic.TOOL_ROUTING,
    KnowledgeTopic.STRUCTURED_OUTPUTS,
    KnowledgeTopic.AGENT_HARDENING,
    KnowledgeTopic.GUARDRAILS,
    KnowledgeTopic.HUMAN_IN_LOOP,
    KnowledgeTopic.IDENTITY_ACCESS,
    KnowledgeTopic.EVALUATION,
    KnowledgeTopic.OBSERVABILITY,
    KnowledgeTopic.COST_LATENCY,
    KnowledgeTopic.CODING_AGENTS,
    KnowledgeTopic.MODEL_ROUTING,
    KnowledgeTopic.DATA_CONNECTORS,
    KnowledgeTopic.KNOWLEDGE_GRAPHS,
    KnowledgeTopic.MULTI_AGENT,
    KnowledgeTopic.COMPUTER_USE,
    KnowledgeTopic.MULTIMODAL_AGENTS,
    KnowledgeTopic.WORKFLOW,
    KnowledgeTopic.SAFETY,
    KnowledgeTopic.DEPLOYMENT,
]

SOURCE_QUALIFIERS = {
    "authority": ["official docs", "specification", "SDK docs", "technical report"],
    "implementation": ["GitHub", "open source", "reference implementation", "example repo"],
    "risk": ["failure modes", "best practices", "security", "evaluation"],
    "blog": ["case study", "production lessons", "technical blog", "postmortem"],
    "chinese": ["中文博客", "技术分享", "实践经验", "案例分析", "知乎", "小红书"],
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
    seed_terms = unique_strings(
        TOPIC_SEED_TERMS.get(topic, [topic.value.replace("_", " ")])
        + CHINESE_TOPIC_TERMS.get(topic, [])
    )
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
    blog_queries = [
        f"{term} {qualifier} {year}"
        for term in seed_terms
        for qualifier in SOURCE_QUALIFIERS["blog"]
    ]
    chinese_queries = [
        f"{term} {qualifier} {year}"
        for term in seed_terms
        for qualifier in SOURCE_QUALIFIERS["chinese"]
    ]
    return TopicExpansion(
        topic=topic.value,
        seed_terms=seed_terms,
        adjacent_terms=unique_strings(adjacent_terms),
        authority_queries=authority_queries,
        implementation_queries=implementation_queries,
        risk_queries=risk_queries + blog_queries + chinese_queries,
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
        f"agentic AI production case study {year}",
        f"AI agent security guardrails tool permissions {year}",
        f"stateful runtime for agents {year}",
        f"human in the loop agent workflow {year}",
        f"AI Agent 大模型智能体 技术趋势 {year}",
        f"大模型 Agent 工程实践 {year}",
        f"智能体 工作流 记忆 工具调用 {year}",
    ]


def build_source_hub_queries(year: int) -> list[str]:
    return [
        f"site:docs.anthropic.com agents {year}",
        f"site:platform.openai.com agents tools memory {year}",
        f"site:langchain.com agent memory evaluation {year}",
        f"site:modelcontextprotocol.io specification {year}",
        f"site:github.com agent framework memory eval {year}",
        f"site:arxiv.org agent memory evaluation tool use {year}",
        f"site:openai.com agents tools runtime {year}",
        f"site:anthropic.com agent tools memory {year}",
        f"site:blog.langchain.com agents evaluation observability {year}",
        f"site:medium.com AI agent production lessons {year}",
        f"site:zhihu.com AI Agent 大模型智能体 {year}",
        f"site:juejin.cn AI Agent 大模型 工程实践 {year}",
        f"site:mp.weixin.qq.com AI Agent MCP RAG 智能体 {year}",
        f"site:bilibili.com AI Agent 智能体 教程 {year}",
    ]


def build_stop_signal_queries(year: int) -> list[str]:
    return [
        f"agent development survey {year}",
        f"AI agents benchmark leaderboard {year}",
        f"agent systems production retrospective {year}",
        f"AI Agent 综述 {year}",
        f"大模型智能体 benchmark {year}",
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
