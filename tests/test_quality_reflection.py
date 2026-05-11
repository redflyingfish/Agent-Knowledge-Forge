from agent_knowledge_harvester.analysis.quality_reflection import (
    parse_quality_reflection,
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
