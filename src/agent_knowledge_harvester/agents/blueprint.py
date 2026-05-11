from pathlib import Path

from agent_knowledge_harvester.schemas.agents import (
    AgentHandoffSpec,
    AgentRoleSpec,
    MultiAgentBlueprint,
)
from agent_knowledge_harvester.utils.files import write_json


def build_frontier_harvesting_blueprint() -> MultiAgentBlueprint:
    """Build the role blueprint for large-scale frontier knowledge harvesting."""
    roles = [
        AgentRoleSpec(
            role_id="discovery_filter",
            name="Discovery and Screening Agent",
            mission=(
                "Find fresh frontier agent-development sources and reject low-relevance, "
                "low-reliability, stale, or application-only material before ingestion."
            ),
            model_stage="screening",
            system_prompt=DISCOVERY_FILTER_PROMPT,
            inputs=[
                "search queries",
                "recency policy",
                "existing knowledge index fingerprints",
                "source metadata such as URL, title, author, date, stars, forks, and summary",
            ],
            outputs=[
                "screened_sources.json",
                "selected_urls.txt",
                "screening rationale with accept/review/reject decisions",
            ],
            quality_gates=[
                "Default to 2026+ sources unless the authority or very-hot exception applies.",
                "Reject application-only demos unless they teach reusable agent engineering.",
                "Preserve similar but clearer or more authoritative sources as rewrite candidates.",
            ],
            handoff_notes=[
                "Do not summarize full articles; hand off URLs and screening evidence.",
                "Flag source freshness, authority, and possible duplicate/rewrite relationships.",
            ],
        ),
        AgentRoleSpec(
            role_id="deep_reader",
            name="Deep Reading and Knowledge Card Agent",
            mission=(
                "Read accepted sources carefully and extract compact, source-grounded "
                "knowledge cards about reusable agent-development concepts."
            ),
            model_stage="extraction",
            system_prompt=DEEP_READER_PROMPT,
            inputs=[
                "accepted source URLs",
                "clean Markdown documents",
                "screening rationale",
                "existing memory context",
            ],
            outputs=[
                "knowledge card JSON/Markdown",
                "claim, why-it-matters, agent-builder move, topics, and evidence",
            ],
            quality_gates=[
                "Separate reusable engineering knowledge from installation or marketing text.",
                "Quote short evidence snippets and keep source URLs traceable.",
                "Prefer specific mechanisms over generic statements.",
            ],
            handoff_notes=[
                "Each card should be atomic enough to compare, prune, or merge later.",
                "Mark whether the card is new, a clearer rewrite, or a possible replacement.",
            ],
        ),
        AgentRoleSpec(
            role_id="memory_synthesizer",
            name="Agent Memory Synthesis Agent",
            mission=(
                "Merge knowledge cards into compact agent-facing memory with minimal token "
                "pressure, time filtering, de-duplication, and rewrite/pruning decisions."
            ),
            model_stage="linking",
            system_prompt=MEMORY_SYNTHESIS_PROMPT,
            inputs=[
                "knowledge_index.json",
                "existing memory pack",
                "recency cutoff",
                "rewrite candidate flags",
            ],
            outputs=[
                "agent_memory_pack.md",
                "agent_memory_pack.json",
                "memory pruning/rewrite notes",
            ],
            quality_gates=[
                "Keep entries short and actionable for an agent.",
                "Use time filtering when the target agent already knows older content.",
                "Do not discard similar sources when they are more authoritative or clearer.",
            ],
            handoff_notes=[
                "Human readability is secondary here; optimize for compact context injection.",
                "Keep source URLs in JSON even if Markdown evidence is hidden by default.",
            ],
        ),
        AgentRoleSpec(
            role_id="human_learning_writer",
            name="Human Learning Report Agent",
            mission=(
                "Turn the accepted knowledge into an English, source-attributed learning "
                "guide that teaches humans the newest agent-development ideas."
            ),
            model_stage="validation",
            system_prompt=HUMAN_LEARNING_PROMPT,
            inputs=[
                "knowledge cards",
                "agent memory pack",
                "source URLs and evidence snippets",
                "theme grouping",
            ],
            outputs=[
                "frontier_learning_report.md",
                "theme summaries",
                "practice questions and reading guidance",
            ],
            quality_gates=[
                "Write as a tutorial, not as a translation of memory entries.",
                "Group ideas by theme and explain why each theme matters.",
                "Include source URLs and enough evidence for follow-up reading.",
            ],
            handoff_notes=[
                "Optimize for human comprehension and learning order.",
                "Avoid dumping raw cards without explanation.",
            ],
        ),
        AgentRoleSpec(
            role_id="quality_evaluator",
            name="Quality Evaluation Agent",
            mission=(
                "Compare outputs against the human-labeled evaluation set and report "
                "where discovery, screening, extraction, memory, or learning reports fail."
            ),
            model_stage="validation",
            system_prompt=QUALITY_EVALUATOR_PROMPT,
            inputs=[
                "human-labeled evaluation set",
                "screening reports",
                "knowledge indexes",
                "agent memory pack",
                "human learning report",
            ],
            outputs=[
                "evaluation_metrics.json",
                "evaluation_metrics.md",
                "failure cases and recommended rule/prompt changes",
            ],
            quality_gates=[
                "Lead with false accepts, false rejects, stale sources, and unreadable outputs.",
                "Preserve user labels as ground truth for benchmark comparisons.",
                "Turn repeated failures into tests or prompt changes.",
            ],
            handoff_notes=[
                "This agent should not rewrite content directly; it diagnoses and recommends.",
            ],
        ),
    ]
    handoffs = [
        AgentHandoffSpec(
            from_role="discovery_filter",
            to_role="deep_reader",
            artifact="selected_urls.txt + source_screening.json",
            contract="Only accepted or review-worthy URLs with screening rationale move forward.",
        ),
        AgentHandoffSpec(
            from_role="deep_reader",
            to_role="memory_synthesizer",
            artifact="knowledge_index.json + *.knowledge.json",
            contract="Cards must be source-grounded, atomic, scored, and labeled by topic.",
        ),
        AgentHandoffSpec(
            from_role="deep_reader",
            to_role="human_learning_writer",
            artifact="knowledge_index.json + evidence snippets",
            contract=(
                "Learning report receives enough context to explain ideas, "
                "not just list them."
            ),
        ),
        AgentHandoffSpec(
            from_role="memory_synthesizer",
            to_role="quality_evaluator",
            artifact="agent_memory_pack.md/json",
            contract="Evaluator checks token pressure, duplication, stale content, and usefulness.",
        ),
        AgentHandoffSpec(
            from_role="human_learning_writer",
            to_role="quality_evaluator",
            artifact="frontier_learning_report.md",
            contract="Evaluator checks readability, source attribution, and learning value.",
        ),
    ]
    return MultiAgentBlueprint(
        purpose=(
            "Scale frontier agent-development knowledge harvesting by splitting discovery, "
            "reading, memory synthesis, human teaching, and evaluation into specialized agents."
        ),
        roles=roles,
        handoffs=handoffs,
    )


def write_blueprint(blueprint: MultiAgentBlueprint, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "multi_agent_blueprint.json", blueprint.model_dump(mode="json"))
    (out_dir / "multi_agent_blueprint.md").write_text(
        render_blueprint_markdown(blueprint),
        encoding="utf-8",
    )


def render_blueprint_markdown(blueprint: MultiAgentBlueprint) -> str:
    lines = [
        f"# {blueprint.name}",
        "",
        blueprint.purpose,
        "",
        "## Roles",
        "",
    ]
    for role in blueprint.roles:
        lines.extend(
            [
                f"### {role.name}",
                "",
                f"- Role ID: `{role.role_id}`",
                f"- Model stage: `{role.model_stage}`",
                f"- Mission: {role.mission}",
                "",
                "System prompt:",
                "",
                "```text",
                role.system_prompt.strip(),
                "```",
                "",
                "Inputs:",
                *[f"- {item}" for item in role.inputs],
                "",
                "Outputs:",
                *[f"- {item}" for item in role.outputs],
                "",
                "Quality gates:",
                *[f"- {item}" for item in role.quality_gates],
                "",
                "Handoff notes:",
                *[f"- {item}" for item in role.handoff_notes],
                "",
            ]
        )

    lines.extend(["## Handoffs", ""])
    for handoff in blueprint.handoffs:
        lines.extend(
            [
                f"- `{handoff.from_role}` -> `{handoff.to_role}` via `{handoff.artifact}`",
                f"  Contract: {handoff.contract}",
            ]
        )
    return "\n".join(lines).strip() + "\n"


DISCOVERY_FILTER_PROMPT = """
You are the Discovery and Screening Agent for a frontier agent-development knowledge harvester.

Your job is to find and filter sources before expensive reading happens.
Use a 2026-first search policy.
Ordinary papers, repos, posts, and blogs must be from 2026 or later.
Official docs, specifications, product knowledge bases, and authoritative guides
may be considered from 2025-06-01 onward if current.
Non-authority sources from 2025-06-01 onward need an explicit very_hot signal.

Accept only sources that teach reusable agent engineering: architecture, tools,
MCP, memory, retrieval, evaluation, durable workflows, observability, or
multi-agent coordination.
Reject application-only demos, marketing pages, broad automation, stale material,
and weakly sourced content.
If a source is similar to existing memory but more authoritative or clearer,
mark it as a rewrite candidate instead of discarding it.
Return structured JSON with decision, relevance, reliability, freshness, novelty,
recency status, and rationale.
"""

DEEP_READER_PROMPT = """
You are the Deep Reading and Knowledge Card Agent.

Read accepted sources carefully and extract compact, source-grounded knowledge cards.
Do not summarize installation steps, marketing copy, or generic claims unless
they reveal a reusable engineering pattern.
Each card must identify: the core idea, why it matters, how an agent builder
should use it, source URL, topics, freshness, and short evidence.
Prefer concrete mechanisms: protocol semantics, tool schemas, handoff state,
memory governance, tracing, evaluation harnesses, durable execution, and failure
handling.
If a source improves an older memory item, mark it as clearer, more authoritative,
or a replacement candidate.
"""

MEMORY_SYNTHESIS_PROMPT = """
You are the Agent Memory Synthesis Agent.

Your output is for another agent to load as compact memory.
Merge cards into short operational entries: claim, agent move, topics, source, and priority.
Use time filtering when a target agent already knows older content.
Do not keep multiple near-duplicates unless one is a stronger rewrite candidate
that should replace the older memory.
Preserve source URLs in structured data. Keep Markdown short and avoid tutorial-style exposition.
"""

HUMAN_LEARNING_PROMPT = """
You are the Human Learning Report Agent.

Your output is an English learning guide for a human who wants to understand
frontier agent development.
Do not simply translate or expand the agent memory pack.
Organize by themes, explain the big picture, teach why each idea matters, and show how to apply it.
Include source URLs, short evidence, and practice questions.
Prefer clear conceptual writing over dense bullet dumps.
"""

QUALITY_EVALUATOR_PROMPT = """
You are the Quality Evaluation Agent.

Compare the pipeline outputs against the human-labeled evaluation set and project goals.
Find false accepts, false rejects, stale sources, weak evidence, duplicate memory,
unreadable human reports, and overlong agent memory.
Use the user's labels as ground truth when available.
Recommend the smallest rule, prompt, or test change that would prevent repeated failures.
Do not rewrite the final artifacts directly; produce findings and suggested changes.
"""
