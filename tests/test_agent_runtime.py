from pydantic import TypeAdapter

from agent_knowledge_harvester.agents.runtime import deduplicate_targets, render_team_trace
from agent_knowledge_harvester.schemas.agents import TeamRunTrace, TeamStageTrace
from agent_knowledge_harvester.schemas.ingestion import SourceKind, UrlTarget


def test_deduplicate_targets_keeps_first_url_variant() -> None:
    adapter = TypeAdapter(UrlTarget)
    targets = [
        adapter.validate_python(
            {"url": "https://example.com/a", "source_kind": SourceKind.URL.value}
        ),
        adapter.validate_python(
            {"url": "https://example.com/a/", "source_kind": SourceKind.URL.value}
        ),
        adapter.validate_python(
            {"url": "https://example.com/b", "source_kind": SourceKind.URL.value}
        ),
    ]

    assert [str(target.url) for target in deduplicate_targets(targets)] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_render_team_trace_lists_stage_metrics_and_outputs() -> None:
    trace = TeamRunTrace(
        run_name="smoke",
        blueprint_name="Frontier Agent Knowledge Harvesting Team",
        output_dir="data/team-run",
        stages=[
            TeamStageTrace(
                role_id="discovery_filter",
                status="completed",
                model_stage="screening",
                output_artifacts=["source_screening.json"],
                metrics={"accepted": 2},
            )
        ],
    )

    rendered = render_team_trace(trace)

    assert "# Multi-Agent Team Run Trace" in rendered
    assert "discovery_filter" in rendered
    assert "accepted: 2" in rendered
    assert "source_screening.json" in rendered
