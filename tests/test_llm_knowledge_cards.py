from agent_knowledge_harvester.analysis.llm_knowledge_cards import (
    analyze_directory_with_llm,
    build_llm_extraction_user_prompt,
    parse_llm_analysis_result,
)
from agent_knowledge_harvester.schemas.analysis import AnalysisResult, KnowledgeCard, KnowledgeTopic
from agent_knowledge_harvester.schemas.ingestion import (
    CleanDocument,
    FetchMethod,
    FetchStats,
    IngestionResult,
    SourceKind,
    UrlTarget,
)
from agent_knowledge_harvester.utils.files import write_json


def make_document() -> CleanDocument:
    return CleanDocument(
        source_url="https://example.com/mcp",
        title="MCP Spec",
        markdown="MCP lets agents connect to tools with JSON-RPC messages.",
        summary_hint="MCP connects agents to tools.",
        technical_signal_score=0.8,
        stats=FetchStats(method=FetchMethod.JINA_READER, source_url="https://example.com/mcp"),
    )


def test_parse_llm_analysis_result_normalizes_cards() -> None:
    result = parse_llm_analysis_result(
        {
            "cards": [
                {
                    "title": "MCP tool boundary",
                    "one_sentence": "MCP gives agents a protocol boundary for tools.",
                    "why_it_matters": "Boundaries make tool use inspectable.",
                    "agent_builder_takeaway": "Define tool contracts outside prompt prose.",
                    "topics": ["mcp", "tool_use", "unknown"],
                    "implementation_notes": ["Keep schemas explicit."],
                    "evidence": ["MCP uses JSON-RPC messages."],
                    "relevance_score": 1.2,
                    "frontier_score": -1,
                }
            ]
        },
        document=make_document(),
        max_cards=4,
    )

    assert len(result.cards) == 1
    assert result.cards[0].topics == [KnowledgeTopic.MCP, KnowledgeTopic.TOOL_USE]
    assert result.cards[0].relevance_score == 1.0
    assert result.cards[0].frontier_score == 0.0


def test_build_llm_extraction_prompt_contains_schema_and_source() -> None:
    prompt = build_llm_extraction_user_prompt(make_document(), max_cards=2, max_markdown_chars=500)

    assert "Source URL: https://example.com/mcp" in prompt
    assert '"cards"' in prompt
    assert "MCP lets agents connect" in prompt


def test_analyze_directory_with_llm_respects_concurrency(tmp_path) -> None:
    class TrackingAnalyzer:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def analyze_ingestion_result(self, result):  # noqa: ANN001
            import asyncio

            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return AnalysisResult(
                source_url=result.target.url,
                title="Concurrency",
                cards=[
                    KnowledgeCard(
                        source_url=result.target.url,
                        title="Concurrent reading",
                        one_sentence="The deep reader can process documents concurrently.",
                        why_it_matters="Concurrency improves broad-run throughput.",
                        agent_builder_takeaway="Use bounded concurrency for LLM extraction.",
                        topics=[KnowledgeTopic.WORKFLOW],
                        relevance_score=0.8,
                        frontier_score=0.5,
                    )
                ],
            )

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for index in range(4):
        target = UrlTarget(
            url=f"https://example.com/{index}",
            source_kind=SourceKind.URL,
        )
        write_json(
            in_dir / f"doc-{index}.json",
            IngestionResult(target=target, success=True, clean=make_document()).model_dump(
                mode="json"
            ),
        )

    analyzer = TrackingAnalyzer()
    import asyncio

    asyncio.run(
        analyze_directory_with_llm(
            in_dir=in_dir,
            out_dir=out_dir,
            analyzer=analyzer,  # type: ignore[arg-type]
            concurrency=2,
        )
    )

    assert analyzer.max_active == 2
    assert (out_dir / "knowledge_index.md").exists()
