from agent_knowledge_harvester.analysis.evaluation import evaluate_outputs, render_evaluation
from agent_knowledge_harvester.schemas.analysis import (
    KnowledgeIndex,
    KnowledgeIndexEntry,
    KnowledgeTopic,
)
from agent_knowledge_harvester.schemas.screening import (
    LLMScreeningJudgment,
    ScreenedSource,
    ScreeningDecision,
    ScreeningReport,
    SourceCandidate,
)


def test_evaluate_outputs_combines_screening_and_knowledge_metrics(tmp_path) -> None:
    screening = ScreeningReport(
        total_candidates=2,
        accepted=1,
        review=0,
        rejected=1,
        sources=[
            ScreenedSource(
                candidate=SourceCandidate(url="https://github.com/a/b", title="a/b"),
                decision=ScreeningDecision.ACCEPT,
                overall_score=0.8,
                relevance_score=0.7,
                authority_score=0.6,
                freshness_score=1.0,
                novelty_score=0.9,
                llm_judgment=LLMScreeningJudgment(
                    decision=ScreeningDecision.ACCEPT,
                    agent_relevance=0.8,
                    reliability=0.7,
                    novelty=0.9,
                    classic_value=0.6,
                    memory_action="ingest",
                ),
            ),
            ScreenedSource(
                candidate=SourceCandidate(url="https://github.com/c/d", title="c/d"),
                decision=ScreeningDecision.REJECT,
                overall_score=0.2,
                relevance_score=0.1,
                authority_score=0.5,
                freshness_score=0.4,
                novelty_score=0.3,
            ),
        ],
    )
    index = KnowledgeIndex(
        total_documents=1,
        total_cards=1,
        topic_counts={KnowledgeTopic.MCP: 1},
        entries=[
            KnowledgeIndexEntry(
                source_url="https://example.com/mcp",
                card_title="MCP",
                one_sentence="MCP connects agents to tools.",
                agent_builder_takeaway="Treat MCP as an integration boundary.",
                topics=[KnowledgeTopic.MCP],
                relevance_score=0.8,
                frontier_score=0.5,
                priority_score=0.7,
                evidence=["MCP connects agents to tools."],
            )
        ],
    )
    markdown_dir = tmp_path / "analysis"
    markdown_dir.mkdir()
    (markdown_dir / "knowledge_index.md").write_text("# Index\n", encoding="utf-8")
    screening_path = tmp_path / "source_screening.json"
    index_path = tmp_path / "knowledge_index.json"
    screening_path.write_text(screening.model_dump_json(), encoding="utf-8")
    index_path.write_text(index.model_dump_json(), encoding="utf-8")

    metrics = evaluate_outputs(screening_path, index_path, markdown_dir)

    assert metrics.screening_accept_rate == 0.5
    assert metrics.avg_source_relevance == 0.4
    assert metrics.llm_judged_sources == 1
    assert metrics.avg_llm_agent_relevance == 0.8
    assert metrics.avg_card_priority == 0.7
    assert metrics.topic_coverage == 1
    assert metrics.evidence_coverage_rate == 1.0
    assert metrics.avg_evidence_per_card == 1.0
    assert metrics.unique_sources == 1
    assert metrics.source_diversity_ratio == 1.0
    assert metrics.max_source_concentration == 1.0
    assert metrics.durable_markdown_chars == len("# Index\n")


def test_render_evaluation_includes_notes() -> None:
    metrics = evaluate_outputs()

    rendered = render_evaluation(metrics)

    assert "screening_report_missing" in rendered
    assert "knowledge_index_missing" in rendered
