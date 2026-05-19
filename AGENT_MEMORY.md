# Agent Memory

This file is for future coding agents working on this repository. It should be concise, operational, and updated when project behavior or evaluation standards change.

## Current System Shape

- Phase 1 ingestion fetches URLs with Jina Reader first, then falls back to Crawl4AI.
- Ingestion outputs raw and clean documents as JSON/Markdown under a caller-provided `data/...` directory.
- Phase 2 analysis converts clean documents into source-grounded knowledge cards.
- Analysis also writes `knowledge_index.md/json`, which ranks cards across a batch by priority.
- Markdown artifacts are preferred for human/agent review; JSON artifacts are kept for structured replay, indexing, MCP/API serving, and tests.

## Important Commands

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/akf ingest --url https://modelcontextprotocol.io/introduction --out data/ingested
.venv/bin/akf trending --language python --since daily --limit 5 --out data/ingested
.venv/bin/akf screen-trending --language python --since daily --limit 10 --out data/screening
.venv/bin/akf screen-trending --language python --since daily --limit 5 --out data/screening-llm --use-llm --llm-max-candidates 5
.venv/bin/akf analyze --in-dir data/ingested --out data/analysis
.venv/bin/akf retrieval-manifest --index data/analysis/knowledge_index.json --out data/analysis
.venv/bin/akf knowledge-clusters --index data/analysis/knowledge_index.json --out data/analysis
.venv/bin/akf memory-pack --analysis-dir data/analysis --out data/memory-pack --after 2026-05-09
.venv/bin/akf agent-blueprint --out data/agent-blueprint
.venv/bin/akf discover --year 2026 --max-queries 40 --results-per-query 5 --out data/discovery-run
.venv/bin/akf run-team --discover --discovery-year 2026 --use-llm-agents --out data/team-run
.venv/bin/akf run-team --trending-language python --trending-limit 10 --out data/team-run
.venv/bin/akf run-team --url https://modelcontextprotocol.io/specification/2025-06-18 --use-llm-agents --out data/team-run-llm
.venv/bin/akf evaluate --screening-report data/screening/source_screening.json --knowledge-index data/analysis/knowledge_index.json --markdown-dir data/analysis
.venv/bin/akf llm-check --stage screening
.venv/bin/akf llm-check --stage screening --ping
.venv/bin/akf cleanup
```

Use `cleanup --yes` only after confirming the listed scratch artifacts have been superseded by analysis outputs.

## Artifact Policy

- Durable review artifacts: `agent.md`, `AGENT_MEMORY.md`, `knowledge_index.md`, and `*.knowledge.md`. A local `ITERATION_LOG.md` may exist during private development, but it is not part of the public repo.
- Machine-readable durable artifacts: `knowledge_index.json`, `*.knowledge.json`, and stats JSON files when they are needed for reproducibility.
- Disposable artifacts: smoke-test raw directories such as `data/smoke`, `data/smoke-trending`, and temporary analysis experiments.
- Do not delete `.venv` during normal iteration; it is large but it is the local runtime environment.

## Known Quality Heuristics

- Preserve code fences during preprocessing.
- Strip Jina Reader wrapper lines before scoring or summarizing.
- Filter GitHub navigation, commit metadata, empty links, file listings, and low-value repository chrome.
- Normalize Markdown tables before sentence scoring.
- Prefer fewer high-signal cards over many noisy cards.
- Compare `knowledge_index.md` before and after tuning filters.
- `knowledge_index.md` should be more than one-line summaries. Keep `Why it matters` and `Implementation details` visible so agents can recover concrete implementation moves without opening every per-source card.
- `knowledge_index.rich.md` is the expanded reading view. It may repeat entries, sources, and topic clusters to make review easier; it is not a compact memory artifact.
- Analysis indexes are durable knowledge-base artifacts and should keep every extracted card by default. Limit them only with `--max-index-entries` for smoke tests or special review lists; context-size compression belongs in memory-pack compact outputs.
- Screening relevance should distinguish core Agent signals from broad automation/browser/testing signals.
- GitHub topics are weak evidence; title and description should carry more weight than self-assigned topics.
- Novelty screening now uses a technical fingerprint in addition to full-token Jaccard and token containment. Shared field terms such as agent, MCP, workflow, tool, and memory should not by themselves make a source look duplicate.
- Treat a source as genuinely more novel when it introduces distinctive technical terms, implementation patterns, evaluation methods, governance constraints, or architecture boundaries.
- Similar sources should not always be discarded. If a later source is more authoritative or clearer, keep it as a possible rewrite/synthesis candidate for improving older memory.

## Baseline Metrics To Track

- Screening: total candidates, accept/review/reject rate, average relevance, authority, freshness, novelty.
- Knowledge: total cards, priority index entries, average priority, average relevance, average frontier score, topic coverage.
- Memory footprint: durable Markdown chars. This approximates token pressure for Agent-readable memory.
- LLM usage: estimated prompt/completion tokens per call once LLM screening/refinement is enabled.
- LLM screening: judged source count, average LLM agent relevance, reliability, novelty, and classic value.
- Memory pack: input entries, retained entries, dropped-by-time count, dropped-by-priority count, duplicate count, Markdown chars.

## Current LLM Screening Behavior

- `.env` is configured for an OpenAI-compatible DeepSeek endpoint.
- `llm-check --stage screening --ping` succeeds with `deepseek-chat`.
- `screen-trending --use-llm` first applies deterministic scoring, then asks the LLM to judge selected candidates as `accept`, `review`, or `reject`.
- Each LLM-reviewed source preserves `pre_llm_decision`, `pre_llm_overall_score`, and `llm_judgment` so future agents can debug rule-vs-LLM disagreements.
- A 2026-05-09 Python Trending smoke run over 5 repos changed the final accept rate from 40% to 20% by rejecting `HKUDS/AI-Trader` as an application-only project rather than reusable Agent engineering knowledge.
- In that run, only `awslabs/aidlc-workflows` remained selected for ingestion.

## Current Memory-Pack Behavior

- `memory-pack` reads one or more `knowledge_index.json` files and writes `agent_memory_pack.md/json` plus `frontier_learning_report.md`.
- `agent_memory_pack.md` is compact Agent-facing memory; `frontier_learning_report.md` is English, source-attributed, and meant for human study.
- Human learning reports should be theme-organized learning guides, not direct translations of memory entries.
- Use `--analysis-dir` for normal analysis output directories or repeated `--index` for explicit files.
- Use `--after <ISO date>` to exclude knowledge indexes generated at or before the Agent's known cutoff time.
- Markdown output hides evidence by default to reduce token pressure; use `--include-evidence` when source snippets are needed.
- A 2026-05-09 smoke pack across `analysis-trending` and `analysis-llm-selected` retained 7 entries: Agent memory was 3160 chars and human report was 6498 chars.
- With `--after 2026-05-09T00:00:00Z`, it retained 4 entries: Agent memory was 1925 chars and human report was 4071 chars.
- Current Jaccard de-duplication is only a coarse lexical guard. Do not treat it as proof that two technical routes are equivalent; repeated core terms are normal in Agent-development material.

## Evaluation Practice

- Curated evaluation sets are local `data/` artifacts and should stay ignored by default.
- Human labels should use `accept`, `review`, or `reject`; short notes should explain whether the item teaches reusable Agent-development knowledge.
- `human_notes` records why the label was chosen. `coverage_notes` records whether the candidate is new, duplicate, more authoritative, better explained, or unclear compared with the current learning report.
- Use frozen local evaluation sets to tune deterministic screening, LLM screening prompts, novelty thresholds, and future pruning rules.
- Future expanded discovery should use a 2025+ broad frontier window: ordinary articles, papers, repos, and technical blogs may enter from 2025 onward when they teach reusable agent engineering.
- Unknown-date non-authority sources need manual review or strong source authority before inclusion.
- Older candidates can remain as legacy evaluation items if already labeled, but should not guide future broad search.

## Research-Informed Design Notes

- Borrow from automated-survey systems when useful: stage discovery, reading, synthesis, and evaluation rather than doing one-shot summaries.
- Prefer schema-validated extraction for LLM-produced knowledge cards, following the same spirit as systematic-review automation work.
- Treat broad-search evaluation as a reproducibility problem: keep frozen local candidate sets, metrics, and next-run plans so runs can be compared.
- Use topic clustering and adjacent-concept expansion to reduce blind spots, but preserve source authority and evidence as stronger signals than topic labels alone.

## Discovery Query Expansion

- `akf query-plan` writes `query_plan.md/json` without doing network search.
- `akf discover` executes the expanded query plan through a configured search provider and writes `candidate_urls.txt`, `search_results.md/json`, and `discovery_stats.json`.
- `run-team --discover` is now the preferred broad-run path. It writes automatic search artifacts under `<run>/00_discovery` before screening.
- Supported search providers are `tavily`, `brave`, `serpapi`, and `exa`, configured through `AKH_SEARCH_PROVIDER` plus either `AKH_SEARCH_API_KEY` or a provider-specific key alias.
- Topic taxonomy now includes newer agent-engineering axes: `agent_hardening`, `context_engineering`, `rag`, `observability`, `coding_agents`, `computer_use`, `protocols`, `structured_outputs`, `safety`, and `deployment`.
- Expanded topic taxonomy also includes `reasoning`, `tool_routing`, `guardrails`, `human_in_loop`, `identity_access`, `state_runtime`, `cost_latency`, `data_connectors`, `knowledge_graphs`, `prompt_engineering`, `model_routing`, and `multimodal_agents`.
- Topics are search seeds, not hard boundaries. Query planning expands each topic into seed terms, adjacent terms, authority queries, implementation queries, and risk/evaluation queries.
- Frontier scout queries are intentionally broad, so the system can discover new concepts outside the current taxonomy.
- Source hub queries target official/spec/repo/paper hubs to reduce dependence on generic keyword search.
- Search result URLs are normalized and deduplicated before screening. Search API results are discovery candidates only; the existing screening stage remains responsible for reliability, relevance, freshness, novelty, and optional LLM judgment.
- 2026-05-15 Tavily smoke discovery succeeded with 5 queries and 15 unique URLs. Real results included useful sources plus marketing/social noise, so deterministic screening must not be bypassed.
- 2026-05-15 full discovery run used 60 Tavily queries, produced 196 unique candidates, selected 40 URLs, ingested 39, and generated 139 LLM-extracted knowledge cards. The full `knowledge_index` was rebuilt to retain all 139 cards instead of only the top 50.
- PDF/resource URLs should not be parsed as HTML during screening. Use URL-derived titles and domain authority for binary sources, then let ingestion/Jina handle deep reading.
- Social posts can contain useful frontier observations, but deterministic screening should keep low-authority social domains in `review` rather than auto-ingesting them.

## Current Multi-Agent Architecture

- `agent-blueprint` writes `data/agent-blueprint/multi_agent_blueprint.md/json`.
- `run-team` is the current end-to-end multi-agent runner. With `--discover`, it writes `00_discovery` first, then staged artifacts under `01_screening`, `02_ingested`, `03_knowledge_base`, `04_memory_packs`, `05_human_report`, `06_evaluation`, plus `team_run_trace.md/json`.
- Prefer `run-team` as the public unified entrypoint. The specialized agents are role stages inside one orchestrated command, not separate user-started daemons.
- Large internet runs must tolerate bad pages: `--ingestion-timeout` / `AKH_INGESTION_TIMEOUT_SECONDS` applies a hard per-URL timeout, and failed pages are persisted as failed ingestion results so the run can continue.
- Control token pressure with `--max-markdown-chars` / `AKH_MAX_MARKDOWN_CHARS`; preprocessing trims long pages before analysis while preserving truncation metadata.
- LLM deep-reader now supports bounded concurrency through `--llm-extraction-concurrency` and records `llm_extraction_concurrency` in stage metrics.
- Recovery path: if a run stops after `02_ingested`, use `analyze --use-llm-extraction --in-dir <run>/02_ingested --out <run>/03_knowledge_base`, then `finalize-run --run-dir <run> --use-llm-agents`.
- The current design has five roles:
- `discovery_filter`: search, recency policy, source reliability, relevance, novelty, rewrite-candidate flags.
- `deep_reader`: careful source reading and source-grounded knowledge cards.
- `memory_synthesizer`: compact Agent-facing memory, time filtering, pruning, rewrite decisions.
- `human_learning_writer`: English human-readable learning guide with themes, explanations, URLs, evidence, and practice questions.
- `quality_evaluator`: compare outputs against the human-labeled evaluation set and identify false accepts, false rejects, stale sources, unreadable reports, and overlong memory.
- Handoffs are explicit artifacts: `selected_urls.txt`, `source_screening.json`, `knowledge_index.json`, `*.knowledge.json`, `agent_memory_pack.md/json`, and `frontier_learning_report.md`.
- Current runner behavior is hybrid by default and LLM-expert when flags are enabled: deterministic stages remain the cheap baseline; screening, extraction, and human report writing can each be upgraded to LLM mode.
- `--use-llm-agents` enables the implemented LLM expert stages: LLM screening, LLM extraction, LLM memory synthesis, LLM human report writing, and LLM quality reflection. Individual flags are `--use-llm-screening`, `--use-llm-extraction`, `--use-llm-memory`, `--use-llm-human-report`, and `--use-llm-reflection`.
- LLM extraction uses `deepseek-reasoner` via the `extraction` stage by default and writes the same `*.knowledge.md/json` plus `knowledge_index.md/json` artifacts as deterministic analysis.
- `03_knowledge_base/retrieval_manifest.md/json` indexes knowledge cards by topic and source.
- `03_knowledge_base/knowledge_chunks.jsonl/json/md` is the RAG/file-search layer. Each chunk preserves text, source URL, topics, scores, evidence, and retrieval-query hints.
- `03_knowledge_base/knowledge_clusters.md/json` is the survey/navigation layer. It groups cards by primary topic and exposes top claims, agent moves, sources, and evidence counts.
- `04_memory_packs` writes uncompressed, working, compact, ultra-compact, and optional LLM-compact memory packs. `agent_memory_pack.uncompressed.md/json` is the long-term memory/RAG/review layer with evidence and no entry cap; compact memory is a distribution layer for direct agent context injection, not the whole knowledge store.
- LLM human report writing uses the `validation` stage. It preserves the deterministic report as `frontier_learning_report.baseline.md`, writes the expert output to `frontier_learning_report.md`, and mirrors it to `frontier_learning_report.llm.md`. The prompt should produce a substantial learning guide, not a compact summary; prefer source names/URLs over bare entry numbers. Large reports may include comparison tables, glossary, reading path, checklist, and exercises.
- LLM quality reflection writes `06_evaluation/quality_reflection.md/json` with next-run suggestions. It should guide future tuning rather than auto-editing rules.
- Evaluation always writes `06_evaluation/next_run_plan.md/json`, converting metrics and optional reflection into concrete next-run adjustments and stop conditions.
- Evaluation metrics include evidence coverage, average evidence per card, unique source count, source diversity ratio, and max source concentration. Use these to detect thin or over-concentrated corpora.
- 2026-05-09 LLM smoke over MCP spec completed 5/5 stages and produced 4 high-quality cards: MCP resource/tool exposure, client-side sampling, three-tier architecture, and consent/control.
- `--use-llm-agents` smoke over MCP spec completed 5/5 stages with screening judged 1, extraction LLM calls 1, fallback count 0, prompt tokens ~=1410, completion tokens ~=988.
- Generic URL screening now fetches lightweight metadata from HTML title/description/date tags. Screening applies domain authority boosts for known official docs/spec domains and writes recency policy reasons into source screening output.
- High-authority official docs/specs with a core Agent/MCP signal can pass via `authority_override_candidate`; this prevents low-keyword official pages from being rejected only because they have sparse summaries.

## Current Best Signals From Smoke Data

- 2026 large-run high-signal engineering patterns: strict tool schemas, sandboxed/truncated tool outputs, entity-isolated memory retrieval, memory decay, durable checkpointing, explicit handoff contracts, human review interrupts, OTel/trace context propagation, and loop/step limits.
- MCP is best treated as an integration boundary between agents, tools, data sources, and workflows.
- The Anthropic financial-services repository suggests a practical package pattern: agents, skills, commands, and MCP connectors.
- Useful Agent-builder takeaways should become implementation moves: define connectors, type tools, preserve evidence, and test handoff state.

## Next Upgrade Candidates

- Use LLM-assisted screening as the default quality comparison path for small evaluation batches, while keeping deterministic screening available as a cheap baseline.
- Add a proper memory store under `memory/` after the Markdown/JSON artifacts stabilize.
- Add an MCP server endpoint that serves `knowledge_index`, `retrieval_manifest`, and `knowledge_chunks`.
- Add recall-oriented search evaluation: track query coverage, source diversity, duplicate rate, failed query rate, and known-good-source recovery for `akf discover`.
- Improve memory-pack pruning with stronger semantic clustering once enough real memory packs exist.
- Consider separating smoke/test artifacts from durable analysis outputs with a stricter directory convention.

## Guardrails For Future Agents

- Read `agent.md` and `AGENT_MEMORY.md` before substantial edits. If a local `ITERATION_LOG.md` exists, use it as private context but do not commit it.
- Do not optimize user-facing summaries unless explicitly requested; prioritize agent-operational memory and retrieval quality.
- Every real-output quality issue should become either a reusable heuristic or a regression test.
- Keep comments and docstrings in English. Keep conversational explanations in Chinese unless asked otherwise.
