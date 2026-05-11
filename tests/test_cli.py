from agent_knowledge_harvester.cli import (
    deduplicate_urls,
    load_urls_from_file,
    normalize_trending_languages,
    runtime_limit_overrides,
)


def test_load_urls_from_file_ignores_blank_lines_and_comments(tmp_path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        """
        # LangGraph docs
        https://langchain-ai.github.io/langgraph/

        https://modelcontextprotocol.io/
        """,
        encoding="utf-8",
    )

    assert load_urls_from_file(url_file) == [
        "https://langchain-ai.github.io/langgraph/",
        "https://modelcontextprotocol.io/",
    ]


def test_deduplicate_urls_keeps_first_seen_order() -> None:
    urls, duplicate_count = deduplicate_urls(
        [
            "https://example.com/a",
            " https://example.com/b ",
            "https://example.com/a",
        ]
    )

    assert urls == ["https://example.com/a", "https://example.com/b"]
    assert duplicate_count == 1


def test_normalize_trending_languages_supports_all_language_search() -> None:
    assert normalize_trending_languages(["python", "all", " "]) == ["python", None]


def test_runtime_limit_overrides_keeps_only_explicit_values() -> None:
    assert runtime_limit_overrides(None, None) == {}
    assert runtime_limit_overrides(30.0, 40_000) == {
        "ingestion_timeout_seconds": 30.0,
        "max_markdown_chars": 40_000,
    }
