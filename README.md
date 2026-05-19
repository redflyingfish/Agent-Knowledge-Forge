# Agent Knowledge Forge

Agent Knowledge Forge is a multi-agent research pipeline for collecting, screening, reading, and packaging frontier AI-agent engineering knowledge. It is designed to turn fast-changing sources such as official docs, specifications, GitHub repositories, papers, and technical articles into:

- source-grounded knowledge cards
- RAG/MCP-friendly retrieval manifests
- embedding-ready JSONL knowledge chunks
- long-term memory packs with evidence
- compact agent-context memory
- human-readable learning reports
- evaluation metrics, quality reflection, and next-run plans

The project focuses on practical agent-development knowledge: memory, RAG, MCP/tool protocols, multi-agent handoffs, planning/reasoning, tool routing, durable state/runtime, observability, coding agents, browser/computer use, context engineering, identity/access, human review, guardrails, cost/latency, safety, and production hardening.

## Why This Exists

Agent frameworks and best practices change quickly. A coding agent often has stale built-in knowledge, while raw web search is noisy and hard to reuse. This project builds a repeatable pipeline that finds recent high-signal sources, filters them, extracts source-grounded patterns, and exports both machine-readable and human-readable knowledge artifacts.

The design is intentionally close to recent automated research and survey systems: staged retrieval/synthesis, schema-checked extraction, evidence-preserving outputs, frozen local evaluation sets, and explicit next-run feedback are treated as first-class engineering requirements.

## Features

- **Unified multi-agent runner**: one command runs automatic search discovery, filtering, deep reading, memory synthesis, human report writing, and evaluation.
- **Provider-neutral search discovery**: executes expanded frontier-agent queries through Tavily, Brave Search, SerpAPI, or Exa, then writes deduplicated URL candidates.
- **LLM expert stages**: optional OpenAI-compatible LLM calls for semantic screening, knowledge-card extraction, compact memory synthesis, human learning reports, and quality reflection.
- **2025+ broad frontier discovery policy**: plans broad searches across recent papers, repos, docs, and technical blogs, then screens candidates with relevance, authority, preview text, and optional LLM judgment.
- **Robust ingestion**: Jina Reader first, Crawl4AI fallback, per-URL hard timeout, Markdown cleanup, and token/character budgets.
- **Source screening**: combines relevance, authority, freshness, novelty, GitHub metadata, and optional LLM judgment.
- **Knowledge cards**: each card records the claim, why it matters, implementation takeaway, topics, scores, source URL, and evidence.
- **RAG-ready chunks**: `knowledge_chunks.jsonl/json/md` preserve claim, source, evidence, topics, scores, and retrieval-query hints for vector stores or file search.
- **Topic clusters**: `knowledge_clusters.md/json` groups cards into survey-style themes for browsing, gap analysis, and next-run planning.
- **Memory layers**:
  - `agent_memory_pack.uncompressed.md/json`: long-term memory/RAG/review layer with evidence.
  - `agent_memory_pack.md/json`: bounded working memory.
  - `agent_memory_pack.compact.md`, `ultra_compact.md`, `llm_compact.md`: direct context-injection layers.
- **Human learning report**: an English study guide with richer themes, source URLs, comparison tables, glossary, checklist, reading path, and practice questions.
- **Evaluation loop**: baseline metrics, evidence/source-diversity checks, optional LLM reflection, and a deterministic next-run plan for tuning discovery, screening, reading, memory, and evaluation.
- **Recovery path**: if a long run stops after ingestion, analysis and finalization can resume from saved artifacts.

## Architecture

```text
search_discovery
  -> query_plan.md/json, search_results.md/json, candidate_urls.txt

discovery_filter
  -> source_screening.json, selected_urls.txt

deep_reader
  -> *.knowledge.md/json, knowledge_index.md/json, retrieval_manifest.md/json,
     knowledge_chunks.jsonl, knowledge_clusters.md/json

memory_synthesizer
  -> uncompressed, working, compact, ultra-compact, and optional LLM compact memory

human_learning_writer
  -> frontier_learning_report.md

quality_evaluator
  -> evaluation_metrics.md/json, quality_reflection.md/json, next_run_plan.md/json
```

The specialist agents are role stages inside one orchestrated command. Users do not need to start separate background agents.

## Installation

```bash
git clone <your-repo-url>
cd agent-knowledge-forge

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m playwright install --with-deps chromium
```

If you use `uv`:

```bash
uv sync --extra dev
bash scripts/install_playwright_deps.sh
```

## LLM Configuration

Copy `.env.example` to `.env` and set an OpenAI-compatible provider.

```bash
AKH_LLM_API_KEY=your_provider_key
AKH_LLM_BASE_URL=https://api.deepseek.com/v1
AKH_LLM_FAST_MODEL=deepseek-chat
AKH_LLM_PRO_MODEL=deepseek-reasoner
```

The fast model is used for screening. The pro model is used for extraction, memory synthesis, reports, and reflection. If both models share the same provider/account, one `AKH_LLM_API_KEY` is enough.

Check configuration without spending tokens:

```bash
.venv/bin/akf llm-check --stage screening
```

Make a tiny test call:

```bash
.venv/bin/akf llm-check --stage screening --ping
```

The legacy `harvester` command is kept as a compatibility alias for `akf`.

## Search Configuration

Automatic discovery uses a web search API. Configure one provider in `.env`:

```bash
AKH_SEARCH_PROVIDER=tavily
AKH_SEARCH_API_KEY=your_search_key
AKH_SEARCH_MAX_QUERIES=80
AKH_SEARCH_RESULTS_PER_QUERY=5
AKH_SEARCH_CONCURRENCY=3
```

Supported providers are `tavily`, `brave`, `serpapi`, and `exa`. You can also use provider-specific key variables such as `AKH_TAVILY_API_KEY`, `AKH_BRAVE_SEARCH_API_KEY`, `AKH_SERPAPI_API_KEY`, or `AKH_EXA_API_KEY`.

## Quick Start

Run automatic discovery only:

```bash
.venv/bin/akf discover \
  --year 2026 \
  --max-queries 40 \
  --results-per-query 5 \
  --out data/discovery-run
```

This writes `candidate_urls.txt`, `search_results.md/json`, `query_plan.md/json`, and `discovery_stats.json`.

Run the complete multi-agent pipeline with automatic discovery:

```bash
.venv/bin/akf run-team \
  --discover \
  --discovery-year 2026 \
  --use-llm-agents \
  --report-language both \
  --llm-max-candidates 25 \
  --max-selected-urls 80 \
  --include-review \
  --concurrency 4 \
  --llm-extraction-concurrency 2 \
  --ingestion-timeout 45 \
  --max-markdown-chars 120000 \
  --out data/broad-run
```

Run the complete pipeline with explicit URLs when you want a tiny reproducible smoke test:

```bash
.venv/bin/akf run-team \
  --url https://modelcontextprotocol.io/specification/2025-06-18 \
  --use-llm-agents \
  --out data/team-run
```

Optionally combine search discovery with GitHub Trending:

```bash
.venv/bin/akf run-team \
  --discover \
  --discovery-year 2026 \
  --trending-language Python \
  --trending-language TypeScript \
  --trending-since weekly \
  --trending-limit 25 \
  --use-llm-agents \
  --llm-max-candidates 25 \
  --max-selected-urls 80 \
  --include-review \
  --concurrency 4 \
  --llm-extraction-concurrency 2 \
  --ingestion-timeout 45 \
  --max-markdown-chars 120000 \
  --out data/broad-run
```

Optionally add your own URL file if you already maintain one locally:

```bash
.venv/bin/akf run-team \
  --url-file data/my_urls.txt \
  --trending-language Python \
  --use-llm-agents \
  --include-review \
  --out data/broad-run
```

`--url-file` is an optional supplement, not a required project artifact. Generated runs and local URL lists belong under `data/`, which is ignored by default.

## MCP Server

Expose a completed run as a read-only MCP knowledge server:

```bash
.venv/bin/akf mcp-server \
  --run-dir data/broad-run
```

The default transport is `stdio`, which is the usual mode for local MCP clients. The server exposes these tools:

- `get_corpus_summary`: inspect corpus size, topics, and top sources.
- `list_topics`: list available topic names and card counts.
- `search_agent_knowledge`: search source-grounded knowledge cards by query, topic, and priority.
- `get_knowledge_card`: fetch the full card returned by search.
- `read_memory_pack`: read working, compact, ultra-compact, LLM-compact, or uncompressed memory packs.
- `read_human_report`: read the generated English, Chinese, or default human report.

Example MCP client command configuration:

```json
{
  "mcpServers": {
    "agent-knowledge-forge": {
      "command": "/absolute/path/to/.venv/bin/akf",
      "args": [
        "mcp-server",
        "--run-dir",
        "/absolute/path/to/data/broad-run"
      ]
    }
  }
}
```

For quick local HTTP inspection:

```bash
.venv/bin/akf mcp-server \
  --run-dir data/broad-run \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000
```

Generate an expanded search plan without network calls:

```bash
.venv/bin/akf query-plan \
  --year 2026 \
  --topic memory \
  --topic rag \
  --topic agent_hardening \
  --out data/query-plan
```

## Recovery Commands

If a long run finishes ingestion but stops before analysis:

```bash
.venv/bin/akf analyze \
  --in-dir data/broad-run/02_ingested \
  --out data/broad-run/03_knowledge_base \
  --use-llm-extraction \
  --llm-extraction-concurrency 2

.venv/bin/akf finalize-run \
  --run-dir data/broad-run \
  --use-llm-agents \
  --report-language both \
  --memory-max-entries 80
```

`knowledge_index.md/json` keeps every extracted card by default. Use `--max-index-entries` only for small smoke tests or intentionally bounded review lists; compact context budgets are handled later by memory-pack outputs.

## Main Outputs

```text
data/<run>/
├── 00_discovery/
│   ├── query_plan.md/json
│   ├── search_results.md/json
│   ├── discovery_stats.json
│   └── candidate_urls.txt
├── 01_screening/
│   ├── source_screening.md/json
│   └── selected_urls.txt
├── 02_ingested/
│   ├── *.md
│   ├── *.json
│   └── run_stats.json
├── 03_knowledge_base/
│   ├── *.knowledge.md/json
│   ├── knowledge_index.md/json
│   ├── knowledge_index.rich.md
│   ├── frontier_brief.md/json
│   ├── retrieval_manifest.md/json
│   ├── knowledge_chunks.md/json/jsonl
│   └── knowledge_clusters.md/json
├── 04_memory_packs/
│   ├── agent_memory_pack.uncompressed.md/json
│   ├── agent_memory_pack.md/json
│   ├── agent_memory_pack.compact.md/json
│   ├── agent_memory_pack.ultra_compact.md/json
│   └── agent_memory_pack.llm_compact.md/json
├── 05_human_report/
│   ├── frontier_learning_report.md
│   ├── frontier_learning_report.en.md
│   └── frontier_learning_report.zh.md
├── 06_evaluation/
│   ├── evaluation_metrics.md/json
│   ├── quality_reflection.md/json
│   └── next_run_plan.md/json
└── team_run_trace.md/json
```

## Evaluation

Compute baseline metrics:

```bash
.venv/bin/akf evaluate \
  --screening-report data/broad-run/01_screening/source_screening.json \
  --knowledge-index data/broad-run/03_knowledge_base/knowledge_index.json \
  --markdown-dir data/broad-run/04_memory_packs \
  --out data/broad-run/06_evaluation
```

Curated evaluation sets can be kept locally under `data/` to calibrate screening and coverage, but they are ignored by default because they often contain local judgments, source snapshots, and paid LLM outputs.

## Development

```bash
.venv/bin/ruff check src/agent_knowledge_harvester tests
.venv/bin/python -m pytest -q
```

Current test coverage includes ingestion timeout behavior, source screening, novelty checks, LLM JSON parsing, LLM knowledge-card extraction, query expansion, multi-agent runtime traces, memory-pack generation, retrieval manifests, human-report prompts, and quality reflection.

## What To Commit

Recommended for open source:

- `src/`
- `tests/`
- `scripts/`
- `pyproject.toml`
- `LICENSE`
- `.env.example`
- `.gitignore`
- `README.md`
- `agent.md`
- `AGENT_MEMORY.md`

Keep ignored:

- `.env`, `.env.*` with real keys
- `.venv/`
- `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`
- `logs/`
- generated `data/*` runs, crawled pages, paid LLM outputs, evaluation sets, and local experiments

## Project Status

This is a research/engineering prototype intended to demonstrate a production-oriented agent knowledge pipeline. It is suitable for experimentation, portfolio review, and further extension into a hosted RAG/MCP knowledge service.

The current open-source pipeline supports automatic search discovery, explicit URLs, optional local URL files, and GitHub Trending. The next natural milestones are stronger recall evaluation, richer connector coverage, and hosted retrieval/MCP serving.
