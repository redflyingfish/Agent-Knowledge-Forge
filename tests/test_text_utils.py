from agent_knowledge_harvester.utils.text import clean_display_text


def test_clean_display_text_repairs_common_mojibake() -> None:
    assert (
        clean_display_text("Use 2\u00e2\u20ac\u201c3 scoped memories\u00e2\u20ac\u00a6")
        == "Use 2-3 scoped memories..."
    )
