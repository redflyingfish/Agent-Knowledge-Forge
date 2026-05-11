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
