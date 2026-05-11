from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from agent_knowledge_harvester.schemas.analysis import KnowledgeIndex
from agent_knowledge_harvester.schemas.screening import ScreeningReport
from agent_knowledge_harvester.utils.files import write_json


class EvaluationMetrics(BaseModel):
    screening_total_candidates: int = 0
    screening_accept_rate: float = 0.0
    screening_review_rate: float = 0.0
    screening_reject_rate: float = 0.0
    avg_source_relevance: float = 0.0
    avg_source_authority: float = 0.0
    avg_source_freshness: float = 0.0
    avg_source_novelty: float = 0.0
    llm_judged_sources: int = 0
    avg_llm_agent_relevance: float = 0.0
    avg_llm_reliability: float = 0.0
    avg_llm_novelty: float = 0.0
    avg_llm_classic_value: float = 0.0
    knowledge_total_cards: int = 0
    knowledge_index_entries: int = 0
    avg_card_priority: float = 0.0
    avg_card_relevance: float = 0.0
    avg_card_frontier: float = 0.0
    topic_coverage: int = 0
    durable_markdown_chars: int = 0
    notes: list[str] = Field(default_factory=list)


def evaluate_outputs(
    screening_report_path: Path | None = None,
    knowledge_index_path: Path | None = None,
    markdown_dir: Path | None = None,
) -> EvaluationMetrics:
    metrics = EvaluationMetrics()
    if screening_report_path and screening_report_path.exists():
        report = ScreeningReport.model_validate_json(
            screening_report_path.read_text(encoding="utf-8")
        )
        apply_screening_metrics(metrics, report)
    else:
        metrics.notes.append("screening_report_missing")

    if knowledge_index_path and knowledge_index_path.exists():
        index = KnowledgeIndex.model_validate_json(knowledge_index_path.read_text(encoding="utf-8"))
        apply_knowledge_index_metrics(metrics, index)
    else:
        metrics.notes.append("knowledge_index_missing")

    if markdown_dir and markdown_dir.exists():
        metrics.durable_markdown_chars = sum(
            len(path.read_text(encoding="utf-8"))
            for path in markdown_dir.glob("*.md")
            if path.is_file()
        )
    else:
        metrics.notes.append("markdown_dir_missing")

    return metrics


def apply_screening_metrics(metrics: EvaluationMetrics, report: ScreeningReport) -> None:
    total = report.total_candidates
    metrics.screening_total_candidates = total
    if total == 0:
        return
    metrics.screening_accept_rate = round(report.accepted / total, 3)
    metrics.screening_review_rate = round(report.review / total, 3)
    metrics.screening_reject_rate = round(report.rejected / total, 3)
    metrics.avg_source_relevance = average(source.relevance_score for source in report.sources)
    metrics.avg_source_authority = average(source.authority_score for source in report.sources)
    metrics.avg_source_freshness = average(source.freshness_score for source in report.sources)
    metrics.avg_source_novelty = average(source.novelty_score for source in report.sources)
    judgments = [source.llm_judgment for source in report.sources if source.llm_judgment]
    metrics.llm_judged_sources = len(judgments)
    metrics.avg_llm_agent_relevance = average(
        judgment.agent_relevance for judgment in judgments
    )
    metrics.avg_llm_reliability = average(judgment.reliability for judgment in judgments)
    metrics.avg_llm_novelty = average(judgment.novelty for judgment in judgments)
    metrics.avg_llm_classic_value = average(judgment.classic_value for judgment in judgments)


def apply_knowledge_index_metrics(metrics: EvaluationMetrics, index: KnowledgeIndex) -> None:
    metrics.knowledge_total_cards = index.total_cards
    metrics.knowledge_index_entries = len(index.entries)
    metrics.topic_coverage = len(index.topic_counts)
    metrics.avg_card_priority = average(entry.priority_score for entry in index.entries)
    metrics.avg_card_relevance = average(entry.relevance_score for entry in index.entries)
    metrics.avg_card_frontier = average(entry.frontier_score for entry in index.entries)


def average(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 3)


def write_evaluation(metrics: EvaluationMetrics, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "evaluation_metrics.json", metrics.model_dump(mode="json"))
    (out_dir / "evaluation_metrics.md").write_text(render_evaluation(metrics), encoding="utf-8")


def render_evaluation(metrics: EvaluationMetrics) -> str:
    lines = [
        "# Evaluation Metrics",
        "",
        "## Screening",
        "",
        f"- total_candidates: {metrics.screening_total_candidates}",
        f"- accept_rate: {metrics.screening_accept_rate:.3f}",
        f"- review_rate: {metrics.screening_review_rate:.3f}",
        f"- reject_rate: {metrics.screening_reject_rate:.3f}",
        f"- avg_source_relevance: {metrics.avg_source_relevance:.3f}",
        f"- avg_source_authority: {metrics.avg_source_authority:.3f}",
        f"- avg_source_freshness: {metrics.avg_source_freshness:.3f}",
        f"- avg_source_novelty: {metrics.avg_source_novelty:.3f}",
        f"- llm_judged_sources: {metrics.llm_judged_sources}",
        f"- avg_llm_agent_relevance: {metrics.avg_llm_agent_relevance:.3f}",
        f"- avg_llm_reliability: {metrics.avg_llm_reliability:.3f}",
        f"- avg_llm_novelty: {metrics.avg_llm_novelty:.3f}",
        f"- avg_llm_classic_value: {metrics.avg_llm_classic_value:.3f}",
        "",
        "## Knowledge",
        "",
        f"- total_cards: {metrics.knowledge_total_cards}",
        f"- index_entries: {metrics.knowledge_index_entries}",
        f"- avg_card_priority: {metrics.avg_card_priority:.3f}",
        f"- avg_card_relevance: {metrics.avg_card_relevance:.3f}",
        f"- avg_card_frontier: {metrics.avg_card_frontier:.3f}",
        f"- topic_coverage: {metrics.topic_coverage}",
        f"- durable_markdown_chars: {metrics.durable_markdown_chars}",
    ]
    if metrics.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in metrics.notes)
    return "\n".join(lines).strip() + "\n"
