from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_knowledge_harvester.analysis.evaluation import EvaluationMetrics
from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex
from agent_knowledge_harvester.schemas.screening import ScreeningReport
from agent_knowledge_harvester.utils.files import write_json


class QualityReflection(BaseModel):
    summary: str = ""
    likely_failure_modes: list[str] = Field(default_factory=list)
    next_run_adjustments: list[str] = Field(default_factory=list)
    prompt_adjustments: list[str] = Field(default_factory=list)
    rule_adjustments: list[str] = Field(default_factory=list)
    risks_to_inspect: list[str] = Field(default_factory=list)


class NextRunPlan(BaseModel):
    objective: str = "Improve frontier-agent knowledge harvesting quality."
    discovery_adjustments: list[str] = Field(default_factory=list)
    screening_adjustments: list[str] = Field(default_factory=list)
    reading_adjustments: list[str] = Field(default_factory=list)
    memory_adjustments: list[str] = Field(default_factory=list)
    evaluation_adjustments: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)


async def write_quality_reflection(
    metrics: EvaluationMetrics,
    screening_report_path: Path,
    knowledge_index_path: Path,
    out_dir: Path,
    llm_client: object,
) -> QualityReflection:
    """Ask the validation expert to reflect on pipeline quality and next actions."""
    report = load_optional_screening_report(screening_report_path)
    index = load_optional_knowledge_index(knowledge_index_path)
    result = await llm_client.chat_json(
        "validation",
        system_prompt=build_quality_reflection_system_prompt(),
        user_prompt=build_quality_reflection_user_prompt(metrics, report, index),
        temperature=0.1,
    )
    reflection = parse_quality_reflection(result.payload)
    write_json(out_dir / "quality_reflection.json", reflection.model_dump(mode="json"))
    (out_dir / "quality_reflection.md").write_text(
        render_quality_reflection(reflection),
        encoding="utf-8",
    )
    return reflection


def write_next_run_plan(
    metrics: EvaluationMetrics,
    out_dir: Path,
    reflection: QualityReflection | None = None,
) -> NextRunPlan:
    """Write a machine-readable plan that converts quality signals into next actions."""
    plan = build_next_run_plan(metrics, reflection)
    write_json(out_dir / "next_run_plan.json", plan.model_dump(mode="json"))
    (out_dir / "next_run_plan.md").write_text(render_next_run_plan(plan), encoding="utf-8")
    return plan


def build_next_run_plan(
    metrics: EvaluationMetrics,
    reflection: QualityReflection | None = None,
) -> NextRunPlan:
    plan = NextRunPlan()
    if metrics.screening_total_candidates == 0:
        plan.discovery_adjustments.append(
            "Add seed URLs or topic-expanded queries before running broad harvesting."
        )
    if metrics.screening_accept_rate < 0.15 and metrics.screening_total_candidates >= 10:
        plan.discovery_adjustments.append(
            "Broaden source hubs and adjacent topic terms; current accept rate is very low."
        )
    if metrics.screening_accept_rate > 0.65:
        plan.screening_adjustments.append(
            "Tighten source screening because the accept rate is high for a frontier corpus."
        )
    if metrics.avg_source_authority and metrics.avg_source_authority < 0.45:
        plan.screening_adjustments.append(
            "Prefer official docs, specifications, high-signal repositories, and papers."
        )
    if metrics.knowledge_index_entries == 0:
        plan.reading_adjustments.append(
            "Inspect ingestion failures and extraction prompts because no indexed cards "
            "were produced."
        )
    if metrics.topic_coverage and metrics.topic_coverage < 4:
        plan.discovery_adjustments.append(
            "Expand topics before the next run; current topic coverage is narrow."
        )
    if metrics.max_source_concentration > 0.35 and metrics.knowledge_index_entries >= 10:
        plan.discovery_adjustments.append(
            "Diversify source hubs because too many cards come from one source."
        )
    if metrics.knowledge_index_entries and metrics.evidence_coverage_rate < 0.75:
        plan.reading_adjustments.append(
            "Tighten extraction so most retained cards include source evidence."
        )
    if metrics.durable_markdown_chars > 120_000:
        plan.memory_adjustments.append(
            "Increase memory pruning or split long-term knowledge into retrieval chunks."
        )
    plan.memory_adjustments.append(
        "Keep uncompressed knowledge for RAG/review and compact memory for direct agent context."
    )
    plan.evaluation_adjustments.extend(
        [
            "Keep a small frozen evaluation set for before/after comparisons.",
            "Check that each retained memory has source URL, evidence, and a concrete agent move.",
        ]
    )
    plan.stop_conditions.extend(
        [
            "Stop broad search when new accepted sources mostly repeat existing "
            "high-priority cards.",
            "Stop a run when source authority, novelty, and topic coverage no longer improve.",
        ]
    )
    if reflection:
        plan.discovery_adjustments.extend(reflection.next_run_adjustments[:3])
        plan.screening_adjustments.extend(reflection.rule_adjustments[:3])
        plan.reading_adjustments.extend(reflection.prompt_adjustments[:3])
        plan.evaluation_adjustments.extend(reflection.risks_to_inspect[:3])
    dedupe_plan(plan)
    return plan


def build_quality_reflection_system_prompt() -> str:
    return (
        "You are a quality reflection expert for an agent-development knowledge harvester. "
        "Use metrics and artifact summaries to identify likely failure modes and concrete "
        "next-run improvements. Do not rewrite artifacts. Return strict JSON only."
    )


def build_quality_reflection_user_prompt(
    metrics: EvaluationMetrics,
    report: ScreeningReport | None,
    index: KnowledgeIndex | None,
) -> str:
    screening_summary = None
    if report:
        screening_summary = [
            {
                "title": source.candidate.title,
                "decision": source.decision.value,
                "overall": source.overall_score,
                "reasons": source.reasons[:6],
            }
            for source in report.sources[:12]
        ]
    knowledge_summary = None
    if index:
        knowledge_summary = [
            {
                "title": entry.card_title,
                "priority": entry.priority_score,
                "topics": [topic.value for topic in entry.topics],
                "source": str(entry.source_url),
            }
            for entry in index.entries[:12]
        ]
    return (
        "Reflect on this run. Identify what might go wrong in broad search and how to improve "
        "the next run. Return JSON keys: summary, likely_failure_modes, next_run_adjustments, "
        "prompt_adjustments, rule_adjustments, risks_to_inspect.\n\n"
        f"Metrics: {metrics.model_dump(mode='json')}\n"
        f"Screening summary: {screening_summary}\n"
        f"Knowledge summary: {knowledge_summary}"
    )


def parse_quality_reflection(payload: dict[str, Any]) -> QualityReflection:
    return QualityReflection(
        summary=str(payload.get("summary") or ""),
        likely_failure_modes=as_string_list(payload.get("likely_failure_modes")),
        next_run_adjustments=as_string_list(payload.get("next_run_adjustments")),
        prompt_adjustments=as_string_list(payload.get("prompt_adjustments")),
        rule_adjustments=as_string_list(payload.get("rule_adjustments")),
        risks_to_inspect=as_string_list(payload.get("risks_to_inspect")),
    )


def render_quality_reflection(reflection: QualityReflection) -> str:
    lines = ["# Quality Reflection", "", reflection.summary or "(no summary)", ""]
    sections = [
        ("Likely Failure Modes", reflection.likely_failure_modes),
        ("Next Run Adjustments", reflection.next_run_adjustments),
        ("Prompt Adjustments", reflection.prompt_adjustments),
        ("Rule Adjustments", reflection.rule_adjustments),
        ("Risks To Inspect", reflection.risks_to_inspect),
    ]
    for title, items in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {item}" for item in items) if items else lines.append("- (none)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_next_run_plan(plan: NextRunPlan) -> str:
    lines = ["# Next Run Plan", "", plan.objective, ""]
    sections = [
        ("Discovery Adjustments", plan.discovery_adjustments),
        ("Screening Adjustments", plan.screening_adjustments),
        ("Reading Adjustments", plan.reading_adjustments),
        ("Memory Adjustments", plan.memory_adjustments),
        ("Evaluation Adjustments", plan.evaluation_adjustments),
        ("Stop Conditions", plan.stop_conditions),
    ]
    for title, items in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {item}" for item in items) if items else lines.append("- (none)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def dedupe_plan(plan: NextRunPlan) -> None:
    for field_name in [
        "discovery_adjustments",
        "screening_adjustments",
        "reading_adjustments",
        "memory_adjustments",
        "evaluation_adjustments",
        "stop_conditions",
    ]:
        setattr(plan, field_name, unique_string_list(getattr(plan, field_name)))


def unique_string_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:400] for item in value if str(item).strip()]


def load_optional_screening_report(path: Path) -> ScreeningReport | None:
    if not path.exists():
        return None
    return ScreeningReport.model_validate_json(path.read_text(encoding="utf-8"))


def load_optional_knowledge_index(path: Path) -> KnowledgeIndex | None:
    if not path.exists():
        return None
    return KnowledgeIndex.model_validate_json(path.read_text(encoding="utf-8"))
