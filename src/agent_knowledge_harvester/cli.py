from pathlib import Path
from typing import Annotated

import typer

from agent_knowledge_harvester.config import settings
from agent_knowledge_harvester.logging import configure_logging

app = typer.Typer(help="Agent Knowledge Forge CLI")


def load_urls_from_file(path: Path) -> list[str]:
    """Read URL candidates from a text file, ignoring blank lines and comments."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def deduplicate_urls(urls: list[str]) -> tuple[list[str], int]:
    """Return URLs in their original order, dropping exact duplicates."""
    seen: set[str] = set()
    unique_urls: list[str] = []

    for url in urls:
        normalized = url.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_urls.append(normalized)

    return unique_urls, len(urls) - len(unique_urls)


@app.command()
def ingest(
    url: Annotated[
        list[str] | None,
        typer.Option("--url", help="URL to ingest. Repeatable."),
    ] = None,
    url_file: Annotated[
        Path | None,
        typer.Option("--url-file", help="Text file with one URL per line."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory."),
    ] = settings.ingested_dir,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, max=8),
    ] = 3,
    ingestion_timeout: Annotated[
        float | None,
        typer.Option("--ingestion-timeout", min=5.0, help="Hard timeout per URL in seconds."),
    ] = None,
    max_markdown_chars: Annotated[
        int | None,
        typer.Option("--max-markdown-chars", min=1000, help="Cleaned Markdown character budget."),
    ] = None,
) -> None:
    """Ingest explicit URLs."""
    configure_logging(settings.logs_dir)
    urls = list(url or [])
    if url_file:
        urls.extend(load_urls_from_file(url_file))

    urls, duplicate_count = deduplicate_urls(urls)

    if not urls:
        raise typer.BadParameter("provide at least one --url or --url-file")

    if duplicate_count:
        typer.echo(f"Skipped {duplicate_count} duplicate URL(s)")

    from agent_knowledge_harvester.ingestion.pipeline import IngestionPipeline

    runtime_settings = settings.model_copy(
        update=runtime_limit_overrides(ingestion_timeout, max_markdown_chars)
    )
    pipeline = IngestionPipeline(runtime_settings, concurrency=concurrency)
    import asyncio

    results = asyncio.run(pipeline.ingest_urls(urls, out_dir=out))
    succeeded = sum(1 for item in results if item.success)
    typer.echo(f"Ingested {succeeded}/{len(results)} URLs into {out}")


@app.command()
def trending(
    language: Annotated[
        str | None,
        typer.Option("--language", "-l"),
    ] = None,
    since: Annotated[
        str,
        typer.Option("--since", help="daily, weekly, or monthly"),
    ] = "daily",
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=25),
    ] = 10,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory."),
    ] = settings.ingested_dir,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, max=8),
    ] = 3,
    ingestion_timeout: Annotated[
        float | None,
        typer.Option("--ingestion-timeout", min=5.0, help="Hard timeout per URL in seconds."),
    ] = None,
    max_markdown_chars: Annotated[
        int | None,
        typer.Option("--max-markdown-chars", min=1000, help="Cleaned Markdown character budget."),
    ] = None,
) -> None:
    """Discover GitHub Trending repositories and ingest them."""
    configure_logging(settings.logs_dir)
    from agent_knowledge_harvester.ingestion.pipeline import IngestionPipeline

    runtime_settings = settings.model_copy(
        update=runtime_limit_overrides(ingestion_timeout, max_markdown_chars)
    )
    pipeline = IngestionPipeline(runtime_settings, concurrency=concurrency)
    import asyncio

    results = asyncio.run(
        pipeline.ingest_github_trending(
            language=language,
            since=since,
            limit=limit,
            out_dir=out,
        )
    )
    succeeded = sum(1 for item in results if item.success)
    typer.echo(f"Ingested {succeeded}/{len(results)} trending repositories into {out}")


@app.command()
def screen_trending(
    language: Annotated[
        str | None,
        typer.Option("--language", "-l"),
    ] = None,
    since: Annotated[
        str,
        typer.Option("--since", help="daily, weekly, or monthly"),
    ] = "daily",
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=25),
    ] = 10,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for screening report."),
    ] = Path("data/screening"),
    knowledge_index: Annotated[
        Path | None,
        typer.Option("--knowledge-index", help="Existing knowledge_index.json for novelty checks."),
    ] = None,
    min_accept_score: Annotated[
        float,
        typer.Option("--min-accept-score", min=0.0, max=1.0),
    ] = 0.5,
    min_relevance_score: Annotated[
        float,
        typer.Option("--min-relevance-score", min=0.0, max=1.0),
    ] = 0.16,
    use_llm: Annotated[
        bool,
        typer.Option("--use-llm/--no-use-llm", help="Refine deterministic screening with LLM."),
    ] = False,
    llm_max_candidates: Annotated[
        int,
        typer.Option("--llm-max-candidates", min=1, max=25),
    ] = 10,
) -> None:
    """Discover and screen GitHub Trending repositories before ingestion."""
    configure_logging(settings.logs_dir)
    import asyncio

    from agent_knowledge_harvester.analysis.source_screening import (
        GitHubRepoMetadataClient,
        load_knowledge_index,
        refine_screening_with_llm,
        screen_candidates,
        write_screening_report,
    )
    from agent_knowledge_harvester.ingestion.github_trending import GitHubTrendingClient
    from agent_knowledge_harvester.llm.client import OpenAICompatibleLLMClient

    async def run() -> None:
        targets = await GitHubTrendingClient(settings).discover(
            language=language,
            since=since,
            limit=limit,
        )
        metadata_client = GitHubRepoMetadataClient(settings)
        candidates = [await metadata_client.enrich(target) for target in targets]
        existing_index = load_knowledge_index(knowledge_index)
        report = screen_candidates(
            candidates,
            existing_index=existing_index,
            min_accept_score=min_accept_score,
            min_relevance_score=min_relevance_score,
        )
        if use_llm:
            llm_client = OpenAICompatibleLLMClient(settings)
            status = llm_client.config_status("screening")
            if not status.configured:
                missing = ",".join(status.missing)
                raise typer.BadParameter(f"LLM screening is not configured; missing={missing}")
            report = await refine_screening_with_llm(
                report,
                llm_client=llm_client,
                existing_index=existing_index,
                stage="screening",
                max_candidates=llm_max_candidates,
            )
        write_screening_report(report, out)
        typer.echo(
            f"Screened {report.total_candidates} sources: "
            f"{report.accepted} accepted, {report.review} review, {report.rejected} rejected. "
            f"LLM judged {report.llm_judged}. "
            f"Selected URLs written to {out / 'selected_urls.txt'}"
        )

    asyncio.run(run())


@app.command()
def analyze(
    in_dir: Annotated[
        Path,
        typer.Option("--in-dir", help="Directory containing ingestion JSON files."),
    ] = settings.ingested_dir,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for knowledge cards."),
    ] = Path("data/analysis"),
    min_relevance: Annotated[
        float,
        typer.Option("--min-relevance", min=0.0, max=1.0, help="Minimum card relevance score."),
    ] = 0.18,
    max_cards_per_doc: Annotated[
        int,
        typer.Option("--max-cards-per-doc", min=1, max=20),
    ] = 4,
    max_index_entries: Annotated[
        int,
        typer.Option("--max-index-entries", min=1, max=200),
    ] = 30,
    use_llm_extraction: Annotated[
        bool,
        typer.Option("--use-llm-extraction/--no-use-llm-extraction"),
    ] = False,
    llm_extraction_concurrency: Annotated[
        int,
        typer.Option("--llm-extraction-concurrency", min=1, max=8),
    ] = 2,
) -> None:
    """Analyze ingested documents into compact knowledge cards."""
    configure_logging(settings.logs_dir)
    import asyncio

    from agent_knowledge_harvester.analysis.knowledge_cards import (
        KnowledgeCardAnalyzer,
        analyze_directory,
    )

    analyzer = KnowledgeCardAnalyzer(
        min_relevance_score=min_relevance,
        max_cards_per_doc=max_cards_per_doc,
    )
    if use_llm_extraction:
        from agent_knowledge_harvester.analysis.llm_knowledge_cards import (
            LLMKnowledgeCardAnalyzer,
            analyze_directory_with_llm,
        )
        from agent_knowledge_harvester.llm.client import OpenAICompatibleLLMClient

        llm_client = OpenAICompatibleLLMClient(settings)
        status = llm_client.config_status("extraction")
        if not status.configured:
            raise typer.BadParameter(
                "LLM extraction is not configured; missing=" + ",".join(status.missing)
            )
        llm_analyzer = LLMKnowledgeCardAnalyzer(
            llm_client=llm_client,
            max_cards_per_doc=max_cards_per_doc,
            fallback_analyzer=analyzer,
        )
        _, stats, index = asyncio.run(
            analyze_directory_with_llm(
                in_dir=in_dir,
                out_dir=out,
                analyzer=llm_analyzer,
                max_index_entries=max_index_entries,
                concurrency=llm_extraction_concurrency,
            )
        )
        typer.echo(
            "Analyzed "
            f"{stats.analyzed_documents}/{stats.total_documents} documents into "
            f"{stats.total_cards} LLM knowledge cards at {out}; "
            f"indexed {len(index.entries)} priority cards; "
            f"llm_calls={llm_analyzer.llm_calls}; "
            f"llm_concurrency={llm_extraction_concurrency}; "
            f"fallbacks={llm_analyzer.fallback_count}"
        )
        return

    _, stats, index = analyze_directory(
        in_dir=in_dir,
        out_dir=out,
        analyzer=analyzer,
        max_index_entries=max_index_entries,
    )
    typer.echo(
        "Analyzed "
        f"{stats.analyzed_documents}/{stats.total_documents} documents into "
        f"{stats.total_cards} knowledge cards at {out}; "
        f"indexed {len(index.entries)} priority cards"
    )


@app.command()
def evaluate(
    screening_report: Annotated[
        Path | None,
        typer.Option("--screening-report", help="Path to source_screening.json."),
    ] = None,
    knowledge_index: Annotated[
        Path | None,
        typer.Option("--knowledge-index", help="Path to knowledge_index.json."),
    ] = None,
    markdown_dir: Annotated[
        Path | None,
        typer.Option("--markdown-dir", help="Directory containing durable Markdown outputs."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for evaluation metrics."),
    ] = Path("data/evaluation"),
) -> None:
    """Compute baseline metrics for screening and knowledge outputs."""
    from agent_knowledge_harvester.analysis.evaluation import evaluate_outputs, write_evaluation

    metrics = evaluate_outputs(
        screening_report_path=screening_report,
        knowledge_index_path=knowledge_index,
        markdown_dir=markdown_dir,
    )
    write_evaluation(metrics, out)
    typer.echo(f"Wrote evaluation metrics to {out}")


@app.command()
def memory_pack(
    index: Annotated[
        list[Path] | None,
        typer.Option("--index", help="Path to a knowledge_index.json file. Repeatable."),
    ] = None,
    analysis_dir: Annotated[
        list[Path] | None,
        typer.Option(
            "--analysis-dir",
            help="Directory containing knowledge_index.json. Repeatable.",
        ),
    ] = None,
    after: Annotated[
        str | None,
        typer.Option("--after", help="Only keep indexes generated after this ISO date/time."),
    ] = None,
    min_priority: Annotated[
        float,
        typer.Option("--min-priority", min=0.0, max=1.0),
    ] = 0.0,
    max_entries: Annotated[
        int,
        typer.Option("--max-entries", min=1, max=200),
    ] = 30,
    include_evidence: Annotated[
        bool,
        typer.Option(
            "--include-evidence/--no-include-evidence",
            help="Include evidence snippets in the Markdown memory pack.",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for compact Agent memory."),
    ] = Path("data/memory-pack"),
) -> None:
    """Build a compact Agent-readable memory pack from knowledge indexes."""
    from agent_knowledge_harvester.memory.pack import (
        build_memory_pack,
        discover_index_paths,
        parse_after,
        write_memory_pack,
    )

    index_paths = list(index or [])
    index_paths.extend(discover_index_paths(list(analysis_dir or [])))
    if not index_paths:
        raise typer.BadParameter("provide at least one --index or --analysis-dir")

    pack = build_memory_pack(
        index_paths=index_paths,
        after=parse_after(after),
        min_priority=min_priority,
        max_entries=max_entries,
    )
    uncompressed_pack = build_memory_pack(
        index_paths=index_paths,
        after=parse_after(after),
        min_priority=min_priority,
        max_entries=None,
        dedupe_threshold=None,
    )
    write_memory_pack(
        pack,
        out,
        include_evidence=include_evidence,
        uncompressed_pack=uncompressed_pack,
    )
    typer.echo(
        f"Wrote {pack.retained_entries}/{pack.total_input_entries} memory entries to {out}: "
        "agent_memory_pack.md, agent_memory_pack.uncompressed.md, compact variants, "
        "and frontier_learning_report.md"
    )


@app.command()
def retrieval_manifest(
    index: Annotated[
        Path,
        typer.Option("--index", help="Path to a knowledge_index.json file."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory for retrieval manifest files."),
    ] = None,
) -> None:
    """Build a RAG/MCP-friendly retrieval manifest from a knowledge index."""
    from agent_knowledge_harvester.memory.retrieval_manifest import write_retrieval_manifest

    manifest = write_retrieval_manifest(index, out_dir=out)
    target_dir = out or index.parent
    typer.echo(f"Wrote {manifest.total_entries} retrieval entries to {target_dir}")


@app.command()
def query_plan(
    topic: Annotated[
        list[str] | None,
        typer.Option("--topic", help="Knowledge topic to expand. Repeatable."),
    ] = None,
    year: Annotated[
        int,
        typer.Option("--year", min=2025, max=2100),
    ] = 2026,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for expanded search queries."),
    ] = Path("data/query-plan"),
) -> None:
    """Generate expanded frontier-agent search queries without doing network search."""
    from agent_knowledge_harvester.discovery.query_expansion import (
        build_query_plan,
        parse_topic_values,
        write_query_plan,
    )

    try:
        topics = parse_topic_values(topic)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    plan = build_query_plan(topics=topics, year=year)
    write_query_plan(plan, out)
    typer.echo(
        f"Wrote {len(plan.topic_expansions)} topic expansion(s), "
        f"{len(plan.frontier_scout_queries)} scout query(s), and "
        f"{len(plan.source_hub_queries)} source-hub query(s) to {out}"
    )


@app.command()
def agent_blueprint(
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for multi-agent role prompts."),
    ] = Path("data/agent-blueprint"),
) -> None:
    """Write the multi-agent harvesting blueprint and role prompts."""
    from agent_knowledge_harvester.agents.blueprint import (
        build_frontier_harvesting_blueprint,
        write_blueprint,
    )

    blueprint = build_frontier_harvesting_blueprint()
    write_blueprint(blueprint, out)
    typer.echo(
        f"Wrote {len(blueprint.roles)} role prompt(s) and "
        f"{len(blueprint.handoffs)} handoff contract(s) to {out}"
    )


@app.command()
def run_team(
    url: Annotated[
        list[str] | None,
        typer.Option("--url", help="Seed URL. Repeatable."),
    ] = None,
    url_file: Annotated[
        Path | None,
        typer.Option("--url-file", help="Text file with one seed URL per line."),
    ] = None,
    trending_language: Annotated[
        list[str] | None,
        typer.Option(
            "--trending-language",
            help="GitHub Trending language to search. Repeatable; use 'all' for no language.",
        ),
    ] = None,
    trending_since: Annotated[
        str,
        typer.Option("--trending-since", help="daily, weekly, or monthly"),
    ] = "daily",
    trending_limit: Annotated[
        int,
        typer.Option("--trending-limit", min=1, max=25),
    ] = 10,
    knowledge_index: Annotated[
        Path | None,
        typer.Option("--knowledge-index", help="Existing knowledge_index.json for novelty checks."),
    ] = None,
    use_llm_screening: Annotated[
        bool,
        typer.Option("--use-llm-screening/--no-use-llm-screening"),
    ] = False,
    use_llm_extraction: Annotated[
        bool,
        typer.Option("--use-llm-extraction/--no-use-llm-extraction"),
    ] = False,
    use_llm_memory: Annotated[
        bool,
        typer.Option("--use-llm-memory/--no-use-llm-memory"),
    ] = False,
    use_llm_human_report: Annotated[
        bool,
        typer.Option("--use-llm-human-report/--no-use-llm-human-report"),
    ] = False,
    use_llm_reflection: Annotated[
        bool,
        typer.Option("--use-llm-reflection/--no-use-llm-reflection"),
    ] = False,
    use_llm_agents: Annotated[
        bool,
        typer.Option("--use-llm-agents/--no-use-llm-agents", help="Enable all LLM expert roles."),
    ] = False,
    llm_max_candidates: Annotated[
        int,
        typer.Option("--llm-max-candidates", min=1, max=25),
    ] = 10,
    max_selected_urls: Annotated[
        int,
        typer.Option("--max-selected-urls", min=1, max=100),
    ] = 20,
    include_review: Annotated[
        bool,
        typer.Option(
            "--include-review/--selected-only",
            help="Let review sources enter ingestion.",
        ),
    ] = False,
    memory_after: Annotated[
        str | None,
        typer.Option("--memory-after", help="Only keep memory entries generated after this date."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for the multi-agent team run."),
    ] = Path("data/team-run"),
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, max=8),
    ] = 3,
    llm_extraction_concurrency: Annotated[
        int,
        typer.Option("--llm-extraction-concurrency", min=1, max=8),
    ] = 2,
    ingestion_timeout: Annotated[
        float | None,
        typer.Option("--ingestion-timeout", min=5.0, help="Hard timeout per URL in seconds."),
    ] = None,
    max_markdown_chars: Annotated[
        int | None,
        typer.Option("--max-markdown-chars", min=1000, help="Cleaned Markdown character budget."),
    ] = None,
) -> None:
    """Run the frontier harvesting multi-agent pipeline end to end."""
    configure_logging(settings.logs_dir)
    import asyncio

    from agent_knowledge_harvester.agents.runtime import FrontierTeamRunner
    from agent_knowledge_harvester.llm.client import OpenAICompatibleLLMClient

    runtime_settings = settings.model_copy(
        update=runtime_limit_overrides(ingestion_timeout, max_markdown_chars)
    )

    seed_urls = list(url or [])
    if url_file:
        seed_urls.extend(load_urls_from_file(url_file))
    seed_urls, duplicate_count = deduplicate_urls(seed_urls)
    languages = normalize_trending_languages(trending_language)
    if not seed_urls and not languages:
        raise typer.BadParameter(
            "provide at least one --url/--url-file or --trending-language"
        )
    if duplicate_count:
        typer.echo(f"Skipped {duplicate_count} duplicate seed URL(s)")
    if use_llm_agents:
        use_llm_screening = True
        use_llm_extraction = True
        use_llm_memory = True
        use_llm_human_report = True
        use_llm_reflection = True
    llm_client = OpenAICompatibleLLMClient(runtime_settings)
    required_stages = []
    if use_llm_screening:
        required_stages.append("screening")
    if use_llm_extraction:
        required_stages.append("extraction")
    if use_llm_memory:
        required_stages.append("linking")
    if use_llm_human_report:
        required_stages.append("validation")
    if use_llm_reflection:
        required_stages.append("validation")
    for stage in required_stages:
        status = llm_client.config_status(stage)
        if not status.configured:
            raise typer.BadParameter(
                f"LLM stage {stage!r} is not configured; missing=" + ",".join(status.missing)
            )

    async def run() -> None:
        trace = await FrontierTeamRunner(runtime_settings).run(
            out_dir=out,
            seed_urls=seed_urls,
            trending_languages=languages,
            trending_since=trending_since,
            trending_limit=trending_limit,
            knowledge_index=knowledge_index,
            use_llm_screening=use_llm_screening,
            llm_max_candidates=llm_max_candidates,
            max_selected_urls=max_selected_urls,
            include_review=include_review,
            ingestion_concurrency=concurrency,
            llm_extraction_concurrency=llm_extraction_concurrency,
            memory_after=memory_after,
            use_llm_extraction=use_llm_extraction,
            use_llm_memory=use_llm_memory,
            use_llm_human_report=use_llm_human_report,
            use_llm_reflection=use_llm_reflection,
        )
        completed = sum(1 for stage in trace.stages if stage.status == "completed")
        typer.echo(
            f"Team run finished with {completed}/{len(trace.stages)} completed stages. "
            f"Trace written to {out / 'team_run_trace.md'}"
        )

    asyncio.run(run())


@app.command()
def finalize_run(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Existing team-run directory to finalize."),
    ] = Path("data/team-run"),
    memory_after: Annotated[
        str | None,
        typer.Option("--memory-after", help="Only keep memory entries generated after this date."),
    ] = None,
    memory_max_entries: Annotated[
        int,
        typer.Option("--memory-max-entries", min=1, max=200),
    ] = 30,
    use_llm_memory: Annotated[
        bool,
        typer.Option("--use-llm-memory/--no-use-llm-memory"),
    ] = False,
    use_llm_human_report: Annotated[
        bool,
        typer.Option("--use-llm-human-report/--no-use-llm-human-report"),
    ] = False,
    use_llm_reflection: Annotated[
        bool,
        typer.Option("--use-llm-reflection/--no-use-llm-reflection"),
    ] = False,
    use_llm_agents: Annotated[
        bool,
        typer.Option("--use-llm-agents/--no-use-llm-agents", help="Enable all final LLM roles."),
    ] = False,
) -> None:
    """Finalize memory, human report, and evaluation from an existing partial run."""
    configure_logging(settings.logs_dir)
    import asyncio

    from agent_knowledge_harvester.agents.runtime import FrontierTeamRunner
    from agent_knowledge_harvester.llm.client import OpenAICompatibleLLMClient

    if use_llm_agents:
        use_llm_memory = True
        use_llm_human_report = True
        use_llm_reflection = True
    llm_client = OpenAICompatibleLLMClient(settings)
    for enabled, stage in [
        (use_llm_memory, "linking"),
        (use_llm_human_report, "validation"),
        (use_llm_reflection, "validation"),
    ]:
        if enabled:
            status = llm_client.config_status(stage)
            if not status.configured:
                raise typer.BadParameter(
                    f"LLM stage {stage!r} is not configured; missing="
                    + ",".join(status.missing)
                )

    async def run() -> None:
        trace = await FrontierTeamRunner(settings).finalize_existing_run(
            out_dir=run_dir,
            memory_after=memory_after,
            memory_max_entries=memory_max_entries,
            use_llm_memory=use_llm_memory,
            use_llm_human_report=use_llm_human_report,
            use_llm_reflection=use_llm_reflection,
        )
        completed = sum(1 for stage in trace.stages if stage.status == "completed")
        typer.echo(
            f"Finalized existing run with {completed}/{len(trace.stages)} completed stages. "
            f"Trace written to {run_dir / 'team_run_trace.md'}"
        )

    asyncio.run(run())


def runtime_limit_overrides(
    ingestion_timeout: float | None = None,
    max_markdown_chars: int | None = None,
) -> dict[str, float | int]:
    """Build per-command runtime setting overrides for crawler safety limits."""
    overrides: dict[str, float | int] = {}
    if ingestion_timeout is not None:
        overrides["ingestion_timeout_seconds"] = ingestion_timeout
    if max_markdown_chars is not None:
        overrides["max_markdown_chars"] = max_markdown_chars
    return overrides


def normalize_trending_languages(values: list[str] | None) -> list[str | None]:
    """Normalize CLI language options while preserving an explicit all-language search."""
    languages: list[str | None] = []
    for value in values or []:
        normalized = value.strip()
        if not normalized:
            continue
        languages.append(None if normalized.lower() == "all" else normalized)
    return languages


@app.command()
def llm_check(
    stage: Annotated[
        str,
        typer.Option("--stage", help="screening, extraction, validation, or linking"),
    ] = "screening",
    ping: Annotated[
        bool,
        typer.Option("--ping", help="Make a tiny JSON LLM call if configured."),
    ] = False,
) -> None:
    """Check whether LLM settings are configured, optionally with a tiny API ping."""
    import asyncio

    from agent_knowledge_harvester.llm.client import OpenAICompatibleLLMClient, normalize_stage

    llm_stage = normalize_stage(stage)
    client = OpenAICompatibleLLMClient(settings)
    status = client.config_status(llm_stage)
    typer.echo(
        f"stage={status.stage} configured={status.configured} "
        f"model={status.model} base_url={status.base_url}"
    )
    if status.missing:
        typer.echo("missing=" + ",".join(status.missing))
    if not ping:
        return

    async def run_ping() -> None:
        result = await client.chat_json(
            llm_stage,
            system_prompt="Return strict JSON only.",
            user_prompt='Return {"ok": true, "purpose": "configuration_check"}.',
        )
        typer.echo(
            "ping_ok=true "
            f"prompt_tokens~={result.prompt_tokens_estimate} "
            f"completion_tokens~={result.completion_tokens_estimate}"
        )

    asyncio.run(run_ping())


@app.command()
def cleanup(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Data directory to inspect."),
    ] = settings.data_dir,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Actually delete the listed disposable artifacts."),
    ] = False,
) -> None:
    """List or remove disposable smoke-test artifacts."""
    from agent_knowledge_harvester.utils.artifacts import find_scratch_artifacts, remove_paths

    artifacts = find_scratch_artifacts(data_dir)
    if not artifacts:
        typer.echo("No disposable smoke-test artifacts found")
        return

    for path in artifacts:
        typer.echo(str(path))

    if not yes:
        typer.echo("Dry run only. Re-run with --yes to delete these artifacts.")
        return

    removed = remove_paths(artifacts)
    typer.echo(f"Removed {removed} disposable artifact path(s)")


@app.command()
def brief(
    index: Annotated[
        Path,
        typer.Option("--index", help="Path to knowledge_index.json."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory for frontier_brief files."),
    ] = None,
) -> None:
    """Generate a compact frontier brief from an existing knowledge index."""
    from agent_knowledge_harvester.analysis.knowledge_cards import write_frontier_brief_from_index

    generated = write_frontier_brief_from_index(index, out_dir=out)
    target_dir = out or index.parent
    typer.echo(
        f"Wrote {generated.title} with {len(generated.top_signals)} signal(s) to {target_dir}"
    )


if __name__ == "__main__":
    app()
