import pytest

from agent_knowledge_harvester.llm.client import parse_json_content


def test_parse_json_content_accepts_strict_json() -> None:
    assert parse_json_content('{"ok": true}') == {"ok": True}


def test_parse_json_content_extracts_fenced_json() -> None:
    assert parse_json_content('```json\n{"ok": true}\n```') == {"ok": True}


def test_parse_json_content_rejects_arrays() -> None:
    with pytest.raises(ValueError):
        parse_json_content("[1, 2, 3]")
