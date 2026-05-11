from agent_knowledge_harvester.agents.blueprint import (
    build_frontier_harvesting_blueprint,
    render_blueprint_markdown,
    write_blueprint,
)


def test_frontier_harvesting_blueprint_defines_specialized_roles() -> None:
    blueprint = build_frontier_harvesting_blueprint()
    role_ids = {role.role_id for role in blueprint.roles}

    assert role_ids == {
        "discovery_filter",
        "deep_reader",
        "memory_synthesizer",
        "human_learning_writer",
        "quality_evaluator",
    }
    assert len(blueprint.handoffs) == 5


def test_role_prompts_encode_recency_and_human_report_rules() -> None:
    blueprint = build_frontier_harvesting_blueprint()
    prompts = {role.role_id: role.system_prompt for role in blueprint.roles}

    assert "2026-first" in prompts["discovery_filter"]
    assert "2025-06-01" in prompts["discovery_filter"]
    assert "Do not simply translate" in prompts["human_learning_writer"]
    assert "human-labeled evaluation set" in prompts["quality_evaluator"]


def test_render_blueprint_markdown_includes_handoff_contracts() -> None:
    rendered = render_blueprint_markdown(build_frontier_harvesting_blueprint())

    assert "# Frontier Agent Knowledge Harvesting Team" in rendered
    assert "Discovery and Screening Agent" in rendered
    assert "Handoffs" in rendered
    assert "selected_urls.txt + source_screening.json" in rendered


def test_write_blueprint_outputs_markdown_and_json(tmp_path) -> None:
    write_blueprint(build_frontier_harvesting_blueprint(), tmp_path)

    assert (tmp_path / "multi_agent_blueprint.md").exists()
    assert (tmp_path / "multi_agent_blueprint.json").exists()
