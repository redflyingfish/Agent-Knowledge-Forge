from agent_knowledge_harvester.analysis.evaluation import EvaluationMetrics
from agent_knowledge_harvester.analysis.quality_reflection import (
    build_next_run_plan,
    parse_quality_reflection,
    render_next_run_plan,
    render_quality_reflection,
)


def test_quality_reflection_normalizes_and_renders_sections() -> None:
    reflection = parse_quality_reflection(
        {
            "summary": "Screening is strict.",
            "likely_failure_modes": ["false rejects"],
            "next_run_adjustments": ["increase official docs coverage"],
            "prompt_adjustments": ["ask for atomic cards"],
            "rule_adjustments": ["boost authority"],
            "risks_to_inspect": ["long memory"],
        }
    )

    rendered = render_quality_reflection(reflection)

    assert "Screening is strict." in rendered
    assert "false rejects" in rendered
    assert "boost authority" in rendered


def test_next_run_plan_turns_metrics_into_actions() -> None:
    metrics = EvaluationMetrics(
        screening_total_candidates=20,
        screening_accept_rate=0.1,
        avg_source_authority=0.3,
        knowledge_index_entries=20,
        topic_coverage=2,
        evidence_coverage_rate=0.5,
        max_source_concentration=0.5,
        durable_markdown_chars=130_000,
    )

    plan = build_next_run_plan(metrics)
    rendered = render_next_run_plan(plan)

    assert "Broaden source hubs" in rendered
    assert "Prefer official docs" in rendered
    assert "Expand topics" in rendered
    assert "Diversify source hubs" in rendered
    assert "include source evidence" in rendered
    assert "Increase memory pruning" in rendered
