from pathlib import Path

from pydantic import HttpUrl, TypeAdapter

from agent_knowledge_harvester.agents.blueprint import build_frontier_harvesting_blueprint
from agent_knowledge_harvester.analysis.evaluation import evaluate_outputs, write_evaluation
from agent_knowledge_harvester.analysis.knowledge_cards import (
    KnowledgeCardAnalyzer,
    analyze_directory,
)
from agent_knowledge_harvester.analysis.llm_knowledge_cards import (
    LLMKnowledgeCardAnalyzer,
    analyze_directory_with_llm,
)
from agent_knowledge_harvester.analysis.quality_reflection import (
    write_next_run_plan,
    write_quality_reflection,
)
from agent_knowledge_harvester.analysis.source_screening import (
    GenericUrlMetadataClient,
    GitHubRepoMetadataClient,
    load_knowledge_index,
    parse_github_repo,
    refine_screening_with_llm,
    screen_candidates,
    write_screening_report,
)
from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.ingestion.github_trending import GitHubTrendingClient
from agent_knowledge_harvester.ingestion.pipeline import IngestionPipeline
from agent_knowledge_harvester.llm.client import OpenAICompatibleLLMClient
from agent_knowledge_harvester.memory.llm_report import render_human_learning_report_with_llm
from agent_knowledge_harvester.memory.llm_synthesis import synthesize_compact_memory_with_llm
from agent_knowledge_harvester.memory.pack import build_memory_pack, parse_after, write_memory_pack
from agent_knowledge_harvester.memory.retrieval_manifest import write_retrieval_manifest
from agent_knowledge_harvester.schemas.agents import TeamRunTrace, TeamStageTrace
from agent_knowledge_harvester.schemas.ingestion import SourceKind, UrlTarget
from agent_knowledge_harvester.schemas.memory import AgentMemoryPack
from agent_knowledge_harvester.schemas.screening import SourceCandidate
from agent_knowledge_harvester.utils.files import write_json


class FrontierTeamRunner:
    """Run the specialized-agent harvesting pipeline with explicit stage handoffs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.blueprint = build_frontier_harvesting_blueprint()

    async def run(
        self,
        out_dir: Path,
        seed_urls: list[str] | None = None,
        trending_languages: list[str | None] | None = None,
        trending_since: str = "daily",
        trending_limit: int = 10,
        knowledge_index: Path | None = None,
        use_llm_screening: bool = False,
        llm_max_candidates: int = 10,
        max_selected_urls: int = 20,
        include_review: bool = False,
        ingestion_concurrency: int = 3,
        llm_extraction_concurrency: int = 2,
        min_relevance: float = 0.18,
        max_cards_per_doc: int = 4,
        max_index_entries: int | None = None,
        memory_after: str | None = None,
        memory_min_priority: float = 0.0,
        memory_max_entries: int = 30,
        use_llm_extraction: bool = False,
        use_llm_memory: bool = False,
        use_llm_human_report: bool = False,
        use_llm_reflection: bool = False,
    ) -> TeamRunTrace:
        out_dir.mkdir(parents=True, exist_ok=True)
        trace = TeamRunTrace(
            run_name=out_dir.name,
            blueprint_name=self.blueprint.name,
            output_dir=str(out_dir),
        )

        screening_dir = out_dir / "01_screening"
        ingest_dir = out_dir / "02_ingested"
        analysis_dir = out_dir / "03_knowledge_base"
        memory_dir = out_dir / "04_memory_packs"
        report_dir = out_dir / "05_human_report"
        evaluation_dir = out_dir / "06_evaluation"

        discovery_stage, selected_urls = await self._run_discovery_filter(
            out_dir=screening_dir,
            seed_urls=seed_urls or [],
            trending_languages=trending_languages or [],
            trending_since=trending_since,
            trending_limit=trending_limit,
            knowledge_index=knowledge_index,
            use_llm_screening=use_llm_screening,
            llm_max_candidates=llm_max_candidates,
            max_selected_urls=max_selected_urls,
            include_review=include_review,
        )
        trace.stages.append(discovery_stage)

        reader_stage = await self._run_deep_reader(
            selected_urls=selected_urls,
            ingest_dir=ingest_dir,
            analysis_dir=analysis_dir,
            ingestion_concurrency=ingestion_concurrency,
            llm_extraction_concurrency=llm_extraction_concurrency,
            min_relevance=min_relevance,
            max_cards_per_doc=max_cards_per_doc,
            max_index_entries=max_index_entries,
            use_llm_extraction=use_llm_extraction,
        )
        trace.stages.append(reader_stage)

        memory_stage = await self._run_memory_synthesis(
            analysis_dir=analysis_dir,
            memory_dir=memory_dir,
            memory_after=memory_after,
            min_priority=memory_min_priority,
            max_entries=memory_max_entries,
            use_llm_memory=use_llm_memory,
        )
        trace.stages.append(memory_stage)

        human_stage = await self._run_human_learning(
            memory_dir,
            report_dir,
            use_llm_human_report=use_llm_human_report,
        )
        trace.stages.append(human_stage)

        quality_stage = await self._run_quality_evaluation(
            screening_dir=screening_dir,
            analysis_dir=analysis_dir,
            memory_dir=memory_dir,
            report_dir=report_dir,
            evaluation_dir=evaluation_dir,
            use_llm_reflection=use_llm_reflection,
        )
        trace.stages.append(quality_stage)

        write_team_trace(trace, out_dir)
        return trace

    async def finalize_existing_run(
        self,
        out_dir: Path,
        memory_after: str | None = None,
        memory_min_priority: float = 0.0,
        memory_max_entries: int = 30,
        use_llm_memory: bool = False,
        use_llm_human_report: bool = False,
        use_llm_reflection: bool = False,
    ) -> TeamRunTrace:
        """Finalize reusable artifacts after a partial team run."""
        out_dir.mkdir(parents=True, exist_ok=True)
        trace = TeamRunTrace(
            run_name=out_dir.name,
            blueprint_name=self.blueprint.name,
            output_dir=str(out_dir),
        )
        screening_dir = out_dir / "01_screening"
        analysis_dir = out_dir / "03_knowledge_base"
        memory_dir = out_dir / "04_memory_packs"
        report_dir = out_dir / "05_human_report"
        evaluation_dir = out_dir / "06_evaluation"

        trace.stages.append(
            await self._run_memory_synthesis(
                analysis_dir=analysis_dir,
                memory_dir=memory_dir,
                memory_after=memory_after,
                min_priority=memory_min_priority,
                max_entries=memory_max_entries,
                use_llm_memory=use_llm_memory,
            )
        )
        trace.stages.append(
            await self._run_human_learning(
                memory_dir,
                report_dir,
                use_llm_human_report=use_llm_human_report,
            )
        )
        trace.stages.append(
            await self._run_quality_evaluation(
                screening_dir=screening_dir,
                analysis_dir=analysis_dir,
                memory_dir=memory_dir,
                report_dir=report_dir,
                evaluation_dir=evaluation_dir,
                use_llm_reflection=use_llm_reflection,
            )
        )
        write_team_trace(trace, out_dir)
        return trace

    async def _run_discovery_filter(
        self,
        out_dir: Path,
        seed_urls: list[str],
        trending_languages: list[str | None],
        trending_since: str,
        trending_limit: int,
        knowledge_index: Path | None,
        use_llm_screening: bool,
        llm_max_candidates: int,
        max_selected_urls: int,
        include_review: bool,
    ) -> tuple[TeamStageTrace, list[str]]:
        role = self._role("discovery_filter")
        stage = TeamStageTrace(role_id=role.role_id, status="running", model_stage=role.model_stage)
        candidates = await self._discover_candidates(
            seed_urls,
            trending_languages,
            trending_since,
            trending_limit,
        )
        existing_index = load_knowledge_index(knowledge_index)
        report = screen_candidates(candidates, existing_index=existing_index)
        if use_llm_screening:
            llm_client = OpenAICompatibleLLMClient(self.settings)
            report = await refine_screening_with_llm(
                report,
                llm_client=llm_client,
                existing_index=existing_index,
                stage="screening",
                max_candidates=llm_max_candidates,
            )

        selected = [
            str(source.candidate.url)
            for source in report.sources
            if source.decision.value == "accept"
            or (include_review and source.decision.value == "review")
        ][:max_selected_urls]
        report.selected_urls = [TypeAdapter(HttpUrl).validate_python(url) for url in selected]
        write_screening_report(report, out_dir)

        stage.status = "completed"
        stage.output_artifacts = [
            str(out_dir / "source_screening.json"),
            str(out_dir / "selected_urls.txt"),
        ]
        stage.metrics = {
            "candidates": report.total_candidates,
            "accepted": report.accepted,
            "review": report.review,
            "rejected": report.rejected,
            "selected_urls": len(selected),
            "llm_judged": report.llm_judged,
        }
        stage.notes = [
            "Applied the 2025+ broad frontier policy before handing URLs to deep reading.",
            (
                "The stage is deterministic by default; LLM screening can be enabled "
                "as a semantic gate."
            ),
        ]
        return stage, selected

    async def _discover_candidates(
        self,
        seed_urls: list[str],
        trending_languages: list[str | None],
        trending_since: str,
        trending_limit: int,
    ) -> list[SourceCandidate]:
        targets: list[UrlTarget] = []
        for url in seed_urls:
            targets.append(
                UrlTarget(
                    url=TypeAdapter(HttpUrl).validate_python(url),
                    source_kind=SourceKind.URL,
                    tags=["seed"],
                    discovered_from="seed_url",
                )
            )

        trending_client = GitHubTrendingClient(self.settings)
        for language in trending_languages:
            targets.extend(
                await trending_client.discover(
                    language=language,
                    since=trending_since,
                    limit=trending_limit,
                )
            )

        return await self._enrich_targets(deduplicate_targets(targets))

    async def _enrich_targets(self, targets: list[UrlTarget]) -> list[SourceCandidate]:
        github_client = GitHubRepoMetadataClient(self.settings)
        generic_client = GenericUrlMetadataClient(self.settings)
        candidates: list[SourceCandidate] = []
        for target in targets:
            owner, repo = parse_github_repo(str(target.url))
            if owner and repo:
                candidates.append(await github_client.enrich(target))
            else:
                candidates.append(await generic_client.enrich(target))
        return candidates

    async def _run_deep_reader(
        self,
        selected_urls: list[str],
        ingest_dir: Path,
        analysis_dir: Path,
        ingestion_concurrency: int,
        llm_extraction_concurrency: int,
        min_relevance: float,
        max_cards_per_doc: int,
        max_index_entries: int | None,
        use_llm_extraction: bool,
    ) -> TeamStageTrace:
        role = self._role("deep_reader")
        stage = TeamStageTrace(role_id=role.role_id, status="running", model_stage=role.model_stage)
        if not selected_urls:
            stage.status = "skipped"
            stage.notes = ["No selected URLs reached the deep reader."]
            return stage

        pipeline = IngestionPipeline(self.settings, concurrency=ingestion_concurrency)
        results = await pipeline.ingest_urls(selected_urls, out_dir=ingest_dir)
        if use_llm_extraction:
            llm_analyzer = LLMKnowledgeCardAnalyzer(
                llm_client=OpenAICompatibleLLMClient(self.settings),
                max_cards_per_doc=max_cards_per_doc,
                fallback_analyzer=KnowledgeCardAnalyzer(
                    min_relevance_score=min_relevance,
                    max_cards_per_doc=max_cards_per_doc,
                ),
            )
            _, stats, index = await analyze_directory_with_llm(
                in_dir=ingest_dir,
                out_dir=analysis_dir,
                analyzer=llm_analyzer,
                max_index_entries=max_index_entries,
                concurrency=llm_extraction_concurrency,
            )
            llm_metrics = {
                "llm_calls": llm_analyzer.llm_calls,
                "llm_extraction_concurrency": llm_extraction_concurrency,
                "llm_prompt_tokens_estimate": llm_analyzer.prompt_tokens_estimate,
                "llm_completion_tokens_estimate": llm_analyzer.completion_tokens_estimate,
                "llm_fallback_count": llm_analyzer.fallback_count,
            }
            if llm_analyzer.fallback_errors:
                llm_metrics["llm_fallback_error"] = llm_analyzer.fallback_errors[0]
        else:
            analyzer = KnowledgeCardAnalyzer(
                min_relevance_score=min_relevance,
                max_cards_per_doc=max_cards_per_doc,
            )
            _, stats, index = analyze_directory(
                in_dir=ingest_dir,
                out_dir=analysis_dir,
                analyzer=analyzer,
                max_index_entries=max_index_entries,
            )
            llm_metrics = {}
        stage.status = "completed"
        stage.input_artifacts = [str(ingest_dir)]
        stage.output_artifacts = [
            str(analysis_dir / "knowledge_index.json"),
            str(analysis_dir / "knowledge_index.md"),
            str(analysis_dir / "knowledge_index.rich.md"),
            str(analysis_dir / "retrieval_manifest.json"),
            str(analysis_dir / "knowledge_chunks.jsonl"),
            str(analysis_dir / "knowledge_clusters.json"),
        ]
        stage.metrics = {
            "selected_urls": len(selected_urls),
            "ingested_success": sum(1 for item in results if item.success),
            "total_documents": stats.total_documents,
            "analyzed_documents": stats.analyzed_documents,
            "knowledge_cards": stats.total_cards,
            "index_entries": len(index.entries),
            "llm_extraction": use_llm_extraction,
            **llm_metrics,
        }
        stage.notes = [
            (
                "Deep reading used the LLM expert extractor."
                if use_llm_extraction
                else "Current deep reading uses deterministic card extraction."
            ),
        ]
        if (analysis_dir / "knowledge_index.json").exists():
            write_retrieval_manifest(analysis_dir / "knowledge_index.json", out_dir=analysis_dir)
        return stage

    async def _run_memory_synthesis(
        self,
        analysis_dir: Path,
        memory_dir: Path,
        memory_after: str | None,
        min_priority: float,
        max_entries: int,
        use_llm_memory: bool,
    ) -> TeamStageTrace:
        role = self._role("memory_synthesizer")
        stage = TeamStageTrace(role_id=role.role_id, status="running", model_stage=role.model_stage)
        index_path = analysis_dir / "knowledge_index.json"
        if not index_path.exists():
            stage.status = "skipped"
            stage.notes = ["No knowledge index was available for memory synthesis."]
            return stage

        pack = build_memory_pack(
            index_paths=[index_path],
            after=parse_after(memory_after),
            min_priority=min_priority,
            max_entries=max_entries,
        )
        uncompressed_pack = build_memory_pack(
            index_paths=[index_path],
            after=parse_after(memory_after),
            min_priority=min_priority,
            max_entries=None,
            dedupe_threshold=None,
        )
        write_memory_pack(pack, memory_dir, uncompressed_pack=uncompressed_pack)
        llm_memory = None
        llm_warning_path = memory_dir / "agent_memory_pack.llm_compact.warning.json"
        if use_llm_memory and pack.entries:
            llm_memory = await synthesize_compact_memory_with_llm(
                pack,
                llm_client=OpenAICompatibleLLMClient(self.settings),
                out_dir=memory_dir,
            )
        stage.status = "completed"
        stage.input_artifacts = [str(index_path)]
        stage.output_artifacts = [
            str(memory_dir / "agent_memory_pack.md"),
            str(memory_dir / "agent_memory_pack.json"),
            str(memory_dir / "agent_memory_pack.uncompressed.md"),
            str(memory_dir / "agent_memory_pack.uncompressed.json"),
            str(memory_dir / "agent_memory_pack.compact.md"),
            str(memory_dir / "agent_memory_pack.ultra_compact.md"),
        ]
        if llm_memory:
            stage.output_artifacts.extend(
                [
                    str(memory_dir / "agent_memory_pack.llm_compact.md"),
                    str(memory_dir / "agent_memory_pack.llm_compact.json"),
                ]
            )
        stage.metrics = {
            "input_entries": pack.total_input_entries,
            "retained_entries": pack.retained_entries,
            "uncompressed_retained_entries": uncompressed_pack.retained_entries,
            "dropped_time": pack.dropped_by_time,
            "dropped_priority": pack.dropped_by_priority,
            "dropped_duplicates": pack.dropped_duplicates,
            "llm_memory_requested": use_llm_memory,
            "llm_memory_success": bool(llm_memory) and not llm_warning_path.exists(),
            "llm_memory_fallback": llm_warning_path.exists(),
        }
        if llm_warning_path.exists():
            stage.output_artifacts.append(str(llm_warning_path))
            stage.notes.append("LLM memory synthesis fell back to deterministic compact memory.")
        return stage

    async def _run_human_learning(
        self,
        memory_dir: Path,
        report_dir: Path,
        use_llm_human_report: bool,
    ) -> TeamStageTrace:
        role = self._role("human_learning_writer")
        report_dir.mkdir(parents=True, exist_ok=True)
        source_report_path = memory_dir / "frontier_learning_report.md"
        report_path = report_dir / "frontier_learning_report.md"
        baseline_path = report_dir / "frontier_learning_report.baseline.md"
        llm_report_path = report_dir / "frontier_learning_report.llm.md"
        warning_path = report_dir / "frontier_learning_report.llm.warning.json"
        if source_report_path.exists() and not report_path.exists():
            report_path.write_text(source_report_path.read_text(encoding="utf-8"), encoding="utf-8")
        llm_report_success = False
        if warning_path.exists():
            warning_path.unlink()
        if report_path.exists() and use_llm_human_report:
            if not baseline_path.exists():
                baseline_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
            pack = AgentMemoryPack.model_validate_json(
                (memory_dir / "agent_memory_pack.json").read_text(encoding="utf-8")
            )
            try:
                markdown = await render_human_learning_report_with_llm(
                    pack,
                    llm_client=OpenAICompatibleLLMClient(self.settings),
                )
                report_path.write_text(markdown, encoding="utf-8")
                llm_report_path.write_text(markdown, encoding="utf-8")
                llm_report_success = True
            except Exception as exc:  # noqa: BLE001 - keep finalization recoverable.
                write_json(
                    warning_path,
                    {
                        "stage": "human_learning_report",
                        "fallback": "baseline_memory_report",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                )
        status = "completed" if report_path.exists() else "skipped"
        return TeamStageTrace(
            role_id=role.role_id,
            status=status,
            model_stage=role.model_stage,
            input_artifacts=[str(memory_dir / "agent_memory_pack.json")],
            output_artifacts=(
                [str(report_path), str(llm_report_path)]
                if llm_report_success and llm_report_path.exists()
                else ([str(report_path)] if report_path.exists() else [])
            )
            + ([str(warning_path)] if warning_path.exists() else []),
            metrics={
                "report_chars": report_path.stat().st_size if report_path.exists() else 0,
                "llm_human_report_requested": use_llm_human_report,
                "llm_human_report_success": llm_report_success,
                "llm_human_report_fallback": warning_path.exists(),
            },
            notes=[
                (
                    "Human report used the LLM learning-writer expert."
                    if llm_report_success
                    else "LLM human report fell back to the baseline memory report."
                    if warning_path.exists()
                    else "Current human report is generated from the memory pack."
                ),
            ],
        )

    async def _run_quality_evaluation(
        self,
        screening_dir: Path,
        analysis_dir: Path,
        memory_dir: Path,
        report_dir: Path,
        evaluation_dir: Path,
        use_llm_reflection: bool,
    ) -> TeamStageTrace:
        role = self._role("quality_evaluator")
        metrics = evaluate_outputs(
            screening_report_path=screening_dir / "source_screening.json",
            knowledge_index_path=analysis_dir / "knowledge_index.json",
            markdown_dir=memory_dir,
        )
        write_evaluation(metrics, evaluation_dir)
        reflection = None
        reflection_warning_path = evaluation_dir / "quality_reflection.warning.json"
        if reflection_warning_path.exists():
            reflection_warning_path.unlink()
        if use_llm_reflection:
            try:
                reflection = await write_quality_reflection(
                    metrics=metrics,
                    screening_report_path=screening_dir / "source_screening.json",
                    knowledge_index_path=analysis_dir / "knowledge_index.json",
                    out_dir=evaluation_dir,
                    llm_client=OpenAICompatibleLLMClient(self.settings),
                )
            except Exception as exc:  # noqa: BLE001 - evaluation metrics still matter.
                write_json(
                    reflection_warning_path,
                    {
                        "stage": "quality_reflection",
                        "fallback": "metrics_only",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                )
        write_next_run_plan(metrics=metrics, out_dir=evaluation_dir, reflection=reflection)
        return TeamStageTrace(
            role_id=role.role_id,
            status="completed",
            model_stage=role.model_stage,
            input_artifacts=[
                str(screening_dir / "source_screening.json"),
                str(analysis_dir / "knowledge_index.json"),
                str(memory_dir),
                str(report_dir),
            ],
            output_artifacts=[
                str(evaluation_dir / "evaluation_metrics.json"),
                str(evaluation_dir / "evaluation_metrics.md"),
                str(evaluation_dir / "next_run_plan.json"),
                str(evaluation_dir / "next_run_plan.md"),
            ]
            + (
                [
                    str(evaluation_dir / "quality_reflection.json"),
                    str(evaluation_dir / "quality_reflection.md"),
                ]
                if reflection
                else []
            ),
            metrics={
                "screening_total": metrics.screening_total_candidates,
                "knowledge_cards": metrics.knowledge_total_cards,
                "markdown_chars": metrics.durable_markdown_chars,
                "llm_reflection_requested": use_llm_reflection,
                "llm_reflection_success": bool(reflection),
                "llm_reflection_fallback": reflection_warning_path.exists(),
            },
            notes=metrics.notes
            + (
                ["LLM quality reflection fell back to metrics-only evaluation."]
                if reflection_warning_path.exists()
                else []
            ),
        )

    def _role(self, role_id: str):
        return next(role for role in self.blueprint.roles if role.role_id == role_id)


def deduplicate_targets(targets: list[UrlTarget]) -> list[UrlTarget]:
    seen: set[str] = set()
    output: list[UrlTarget] = []
    for target in targets:
        key = str(target.url).rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        output.append(target)
    return output


def write_team_trace(trace: TeamRunTrace, out_dir: Path) -> None:
    write_json(out_dir / "team_run_trace.json", trace.model_dump(mode="json"))
    (out_dir / "team_run_trace.md").write_text(render_team_trace(trace), encoding="utf-8")


def render_team_trace(trace: TeamRunTrace) -> str:
    lines = [
        "# Multi-Agent Team Run Trace",
        "",
        f"- Run: {trace.run_name}",
        f"- Blueprint: {trace.blueprint_name}",
        f"- Output directory: {trace.output_dir}",
        "",
        "## Stages",
        "",
    ]
    for stage in trace.stages:
        lines.extend(
            [
                f"### {stage.role_id}",
                "",
                f"- Status: {stage.status}",
                f"- Model stage: {stage.model_stage or '(none)'}",
            ]
        )
        if stage.metrics:
            lines.append("- Metrics:")
            lines.extend(f"  - {key}: {value}" for key, value in stage.metrics.items())
        if stage.output_artifacts:
            lines.append("- Outputs:")
            lines.extend(f"  - {artifact}" for artifact in stage.output_artifacts)
        if stage.notes:
            lines.append("- Notes:")
            lines.extend(f"  - {note}" for note in stage.notes)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
