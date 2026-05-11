# Agent Knowledge Forge

Agent Knowledge Forge is a multi-agent research pipeline for collecting, screening, reading, and packaging frontier AI-agent engineering knowledge. It is designed to turn fast-changing sources such as official docs, specifications, GitHub repositories, papers, and technical articles into:

- source-grounded knowledge cards
- RAG/MCP-friendly retrieval manifests
- long-term memory packs with evidence
- compact agent-context memory
- human-readable learning reports
- evaluation metrics and LLM quality reflection

The project focuses on practical agent-development knowledge: memory, RAG, MCP/tool protocols, multi-agent handoffs, durable execution, observability, coding agents, context engineering, safety, and production hardening.

## Why This Exists

Agent frameworks and best practices change quickly. A coding agent often has stale built-in knowledge, while raw web search is noisy and hard to reuse. This project builds a repeatable pipeline that finds recent high-signal sources, filters them, extracts source-grounded patterns, and exports both machine-readable and human-readable knowledge artifacts.

## Features

- **Unified multi-agent runner**: one command runs discovery, filtering, deep reading, memory synthesis, human report writing, and evaluation.
- **LLM expert stages**: optional OpenAI-compatible LLM calls for semantic screening, knowledge-card extraction, compact memory synthesis, human learning reports, and quality reflection.
- **2026-first discovery policy**: prioritizes current frontier sources while allowing official/spec authority exceptions.
- **Robust ingestion**: Jina Reader first, Crawl4AI fallback, per-URL hard timeout, Markdown cleanup, and token/character budgets.
- **Source screening**: combines relevance, authority, freshness, novelty, GitHub metadata, and optional LLM judgment.
- **Knowledge cards**: each card records the claim, why it matters, implementation takeaway, topics, scores, source URL, and evidence.
- **Memory layers**:
  - `agent_memory_pack.uncompressed.md/json`: long-term memory/RAG/review layer with evidence.
  - `agent_memory_pack.md/json`: bounded working memory.
  - `agent_memory_pack.compact.md`, `ultra_compact.md`, `llm_compact.md`: direct context-injection layers.
- **Human learning report**: an English study guide with themes, source URLs, glossary, checklist, and practice questions.
- **Evaluation loop**: baseline metrics plus optional LLM reflection for next-run tuning.
- **Recovery path**: if a long run stops after ingestion, analysis and finalization can resume from saved artifacts.

## Architecture

```text
discovery_filter
  -> source_screening.json, selected_urls.txt

deep_reader
  -> *.knowledge.md/json, knowledge_index.md/json, retrieval_manifest.md/json

memory_synthesizer
  -> uncompressed, working, compact, ultra-compact, and optional LLM compact memory

human_learning_writer
  -> frontier_learning_report.md

quality_evaluator
  -> evaluation_metrics.md/json, quality_reflection.md/json
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

## Quick Start

Run the complete pipeline with explicit URLs:

```bash
.venv/bin/akf run-team \
  --url https://modelcontextprotocol.io/specification/2025-06-18 \
  --use-llm-agents \
  --out data/team-run
```

Run a broader campaign from a URL file and GitHub Trending:

```bash
.venv/bin/akf run-team \
  --url-file data/seed_urls.txt \
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

Generate an expanded search plan without network calls:

```bash
.venv/bin/akf query-plan \
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
  --llm-extraction-concurrency 2 \
  --max-index-entries 100

.venv/bin/akf finalize-run \
  --run-dir data/broad-run \
  --use-llm-agents \
  --memory-max-entries 80
```

## Main Outputs

```text
data/<run>/
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
│   ├── frontier_brief.md/json
│   └── retrieval_manifest.md/json
├── 04_memory_packs/
│   ├── agent_memory_pack.uncompressed.md/json
│   ├── agent_memory_pack.md/json
│   ├── agent_memory_pack.compact.md/json
│   ├── agent_memory_pack.ultra_compact.md/json
│   └── agent_memory_pack.llm_compact.md/json
├── 05_human_report/
│   └── frontier_learning_report.md
├── 06_evaluation/
│   ├── evaluation_metrics.md/json
│   └── quality_reflection.md/json
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
