from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.ingestion.preprocess import (
    TRUNCATION_MARKER,
    MarkdownPreprocessor,
    collapse_repeated_navigation,
    extract_image_urls,
    extract_jina_reader_title,
    extract_markdown_tables,
    remove_noise_lines,
    strip_jina_reader_preamble,
    technical_signal_score,
    trim_to_budget,
)
from agent_knowledge_harvester.schemas.ingestion import FetchMethod, FetchStats, RawDocument


def test_technical_signal_score_detects_agent_content() -> None:
    markdown = """
    # LangGraph Agent Workflow

    This guide covers multi-agent orchestration, tool calling, MCP, and memory.

    ```python
    graph.invoke({"messages": []})
    ```
    """

    assert technical_signal_score(markdown) > 0.2


def test_preprocessor_removes_noise_and_keeps_source_content() -> None:
    raw = RawDocument(
        source_url="https://example.com/langgraph",
        method=FetchMethod.JINA_READER,
        markdown="""
        # LangGraph Notes

        Sign in
        Subscribe

        Use LangGraph for durable agent workflows.
        """,
        stats=FetchStats(method=FetchMethod.JINA_READER, source_url="https://example.com/langgraph"),
    )

    clean = MarkdownPreprocessor(Settings()).clean(raw)

    assert "Sign in" not in clean.markdown
    assert "Subscribe" not in clean.markdown
    assert "durable agent workflows" in clean.markdown
    assert clean.metadata["preprocess_version"] == "phase1-v2"
    assert clean.metadata["was_truncated"] is False


def test_extract_image_urls_keeps_remote_markdown_images_only() -> None:
    urls = extract_image_urls(
        "![Architecture](https://example.com/arch.png) "
        "![Local](./local.png) "
        "![Trace](http://example.com/trace.jpg)"
    )

    assert urls == ["https://example.com/arch.png", "http://example.com/trace.jpg"]


def test_extract_markdown_tables_keeps_compact_source_tables() -> None:
    tables = extract_markdown_tables(
        """
        Intro text.

        | Feature | Agent impact |
        | --- | --- |
        | Checkpoints | Resume long workflows |
        | Guardrails | Block unsafe tool calls |

        Outro text.
        """
    )

    assert len(tables) == 1
    assert "| Feature | Agent impact |" in tables[0]
    assert "| Guardrails | Block unsafe tool calls |" in tables[0]


def test_noise_removal_preserves_code_fence_content() -> None:
    markdown = """
    # Example

    Sign in

    ```text
    Sign in
    Sign in
    ```
    """

    cleaned = remove_noise_lines(markdown)

    assert "\n    Sign in\n\n" not in cleaned
    assert cleaned.count("Sign in") == 2


def test_repeated_navigation_collapse_preserves_code_fence_content() -> None:
    markdown = """
    Docs
    Docs

    ```python
    print("retry")
    print("retry")
    ```
    """

    collapsed = collapse_repeated_navigation(markdown)

    assert collapsed.count("Docs") == 1
    assert collapsed.count('print("retry")') == 2


def test_trim_to_budget_marks_truncated_content() -> None:
    markdown = "a" * 80 + "\n" + "b" * 80

    trimmed = trim_to_budget(markdown, max_chars=60)

    assert TRUNCATION_MARKER in trimmed
    assert trimmed.startswith("a")
    assert trimmed.endswith("b" * 17)


def test_jina_reader_preamble_is_extracted_and_removed() -> None:
    markdown = """
    Title: What is MCP?

    URL Source: https://example.com/mcp

    Markdown Content:
    MCP connects agents to tools and data sources.
    """

    assert extract_jina_reader_title(markdown) == "What is MCP?"
    assert strip_jina_reader_preamble(markdown) == (
        "MCP connects agents to tools and data sources."
    )


def test_preprocessor_removes_leading_llms_documentation_index() -> None:
    raw = RawDocument(
        source_url="https://example.com/mcp",
        method=FetchMethod.JINA_READER,
        markdown="""
        Title: What is MCP?

        URL Source: https://example.com/mcp

        Markdown Content:
        > ## Documentation Index
        >
        > Fetch the complete documentation index at: https://example.com/llms.txt

        MCP connects AI agents to tools, memory, and workflows.
        """,
        stats=FetchStats(method=FetchMethod.JINA_READER, source_url="https://example.com/mcp"),
    )

    clean = MarkdownPreprocessor(Settings()).clean(raw)

    assert clean.title == "What is MCP?"
    assert "Documentation Index" not in clean.markdown
    assert clean.markdown.startswith("MCP connects AI agents")
