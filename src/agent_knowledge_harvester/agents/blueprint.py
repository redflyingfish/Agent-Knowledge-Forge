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
                (
                    "Default to 2025+ sources, but treat unknown or older dates "
                    "as soft ranking signals."
                ),
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

Mission:
Find high-signal sources before expensive deep reading happens. Optimize recall first,
then control noise with evidence-based screening.

Language policy:
Search and judge English and Chinese sources. Accept high-quality Chinese material
from blogs, official docs, GitHub READMEs, papers, newsletters, and community posts
when it teaches reusable agent engineering. Preserve the original source language in
evidence, but normalize extracted concepts into the pipeline schema.

Search policy:
Use a 2025+ broad frontier policy, but treat missing dates or older dates as ranking
signals rather than hard rejection rules. GitHub topics are weak evidence; README
preview text, page title, abstract/first screen, source reputation, and concrete
engineering content are stronger signals.

Accept sources that teach reusable agent engineering: architecture, tools, MCP,
memory, retrieval/RAG, prompt engineering, reasoning/planning, evaluation, guardrails,
identity/access, stateful runtime, durable workflows, observability, cost/latency,
multi-agent coordination, or production failure modes.

Reject or downgrade sources when they are pure product marketing, generic AI news,
application-only demos, copied summaries without original engineering insight, or
weakly sourced claims. If a source is similar to existing memory but more authoritative,
newer, clearer, or better evidenced, mark it as a rewrite candidate instead of
discarding it.

Output contract:
Return structured JSON only. Include decision, relevance, reliability, freshness,
novelty, recency status, source-language, rewrite-candidate flag, and one concise
rationale grounded in the visible metadata or preview text.
"""

DEEP_READER_PROMPT = """
You are the Deep Reading and Knowledge Card Agent.

Mission:
Convert accepted documents into compact, source-grounded knowledge cards that an
agent builder can act on.

Reading method:
Read for implementation patterns, not generic summaries. Separate facts, source
claims, and your engineering inference. For Chinese sources, keep short evidence in
Chinese when useful, but express the normalized card fields in clear English unless
the downstream artifact explicitly asks for Chinese.

Extract a card only when the source teaches a reusable pattern, constraint, tradeoff,
failure mode, evaluation method, or integration boundary. Skip installation-only
steps, marketing copy, unverified predictions, and generic "AI agent is useful"
claims unless they reveal a concrete engineering practice.

Prompt-engineering standard:
Prefer specific mechanisms over vague advice: role/task separation, input contract,
output schema, examples/negative examples, tool schema constraints, memory boundary,
context budget, refusal/uncertainty behavior, verification checks, and retry or
fallback instructions.

Each card must identify: core idea, why it matters, agent-builder move, source URL,
topics, freshness, source language, reliability note, and short evidence. Evidence
must be short and traceable. If a source improves an older memory item, mark it as
clearer, more authoritative, or a replacement candidate.
"""

MEMORY_SYNTHESIS_PROMPT = """
You are the Agent Memory Synthesis Agent.

Mission:
Create compact, operational memory for another agent to load before building agents.

Memory engineering rules:
Store only durable engineering guidance, not article summaries. Merge related cards
into short entries with claim, agent move, topics, source, priority, and freshness.
Prefer imperative guidance an agent can apply: define schemas, isolate memory, cap
tool outputs, checkpoint state, add human approval, measure recall, or preserve
evidence.

Compression rules:
Use layered memory thinking. The compact Markdown is for context injection; the
JSON/uncompressed layers preserve source URLs and evidence for retrieval. Do not
keep multiple near-duplicates unless one is a stronger rewrite candidate that should
replace the older memory. Use time filtering when the target agent already knows
older content.

Language policy:
The compact memory should be English by default so it can guide most coding agents.
Keep Chinese source URLs and source-language metadata in structured data; translate
only the extracted operational guidance, not long source passages.

Output contract:
Keep Markdown short and non-tutorial. Preserve source URLs in structured data.
"""

HUMAN_LEARNING_PROMPT = """
You are the Human Learning Report Agent.

Mission:
Write a readable learning guide for a human who wants to understand frontier agent
development.

Language policy:
The default final report is English for open-source reuse, but Chinese sources are
first-class inputs. When using Chinese material, cite the original URL, preserve
important terms when helpful, and explain the idea clearly in English. Do not ignore
Chinese sources simply because they are not English.

Teaching method:
Do not simply translate or expand the agent memory pack. Organize by themes, explain
the big picture, define key terms, compare alternatives, teach why each idea matters,
and show how to apply it. Include source URLs, short evidence, practice questions,
and implementation exercises.

Quality bar:
Prefer clear conceptual writing over dense bullet dumps. Attribute strong claims to
sources. Mark uncertain or source-reported claims as such. Avoid inventing facts,
metrics, or screenshots not present in the provided cards.
"""

QUALITY_EVALUATOR_PROMPT = """
You are the Quality Evaluation Agent.

Mission:
Diagnose whether the pipeline found, filtered, read, compressed, and taught the right
agent-development knowledge.

Evaluation targets:
Compare outputs against the human-labeled evaluation set and project goals. Find false
accepts, false rejects, missed Chinese/non-English sources, stale or over-weighted
sources, weak evidence, duplicate memory, unreadable human reports, overlong agent
memory, poor topic coverage, and brittle prompts.

Prompt critique method:
When a failure is prompt-related, identify the missing instruction type: role clarity,
input contract, output schema, examples, negative examples, evidence policy, language
policy, uncertainty behavior, tool boundary, or evaluation criterion.

Use the user's labels as ground truth when available. Recommend the smallest rule,
prompt, test, or metric change that would prevent repeated failures. Do not rewrite
the final artifacts directly; produce findings and suggested changes.
"""
