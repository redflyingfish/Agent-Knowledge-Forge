# Agent Collaboration Guide

This project should be developed in a teaching-assistant style. The coding agent is expected to implement changes while explaining the reasoning behind them, so the user can learn how the system is built and why each decision is made.

## Teaching-Assistant Development Mode

- Explain what project context is being inspected before making changes, including why those files or modules matter.
- For substantial project work, read `README.md` first. Treat any local notes or generated run artifacts as private unless they are intentionally documented for release.
- When proposing or implementing a change, describe the implementation approach and the reason for choosing it.
- Prefer small, verifiable improvements that align with the existing architecture before introducing larger abstractions.
- While editing, call out important trade-offs such as correctness, maintainability, testability, performance, and future extensibility.
- After implementation, summarize what changed, how it was verified, and any remaining limitations or next-step suggestions.
- For substantial optimizations, code changes, pipeline runs, or quality investigations, summarize the intent, validation, and observed metric changes in the final response or a tracked release note when appropriate.
- Keep conversational explanations in Chinese unless the user asks otherwise.
- Write basic in-code comments, docstrings, and developer-facing code notes in English for consistency with the codebase.

## Engineering Expectations

- Read `README.md`, `pyproject.toml`, and the relevant source/test files before changing behavior.
- Preserve the existing `src/` package layout and favor local project patterns over unrelated new frameworks.
- Treat the user's implementation suggestions as useful hypotheses, not binding architecture. Use current evidence, project goals, and testable behavior to choose the implementation.
- Keep the core optimization direction centered on: discovering current frontier Agent-development sources, screening for reliability/relevance/novelty, distilling compact Agent-readable memory, and pruning stale low-value memory.
- Keep ingestion behavior source-attributable: raw URLs, fetch method, cleaned Markdown, stats, and metadata should remain traceable.
- Treat preprocessing conservatively. Remove obvious page chrome, but avoid deleting source content, examples, code blocks, or technical details.
- Treat analysis as a compact knowledge-distillation stage. Prefer source-grounded cards over long summaries.
- The minimum useful agent knowledge card should explain: what the idea is, why it matters, how an agent builder should use it, relevant topics, quality scores, and short source evidence.
- Maintain a cross-document priority index when analyzing batches, so the project can answer "what should I read first?" rather than only "what did I collect?"
- Preserve a RAG/file-search distribution layer. `knowledge_chunks.jsonl` should expose compact, source-grounded chunks with URL, topics, scores, evidence, and retrieval-query hints.
- Preserve a survey/navigation layer. `knowledge_clusters.md/json` should group cards by topic so humans and agents can inspect coverage gaps before the next run.
- Keep durable human/Agent memory in Markdown whenever practical. Keep JSON when it is a machine-readable cache, reproducible intermediate, or API substrate.
- For Agent-facing memory, prefer `agent_memory_pack.md` over raw analysis indexes when the goal is compact context injection.
- For long-term memory, RAG, or review, preserve `agent_memory_pack.uncompressed.md/json`; do not confuse long-term storage with context-injection compression.
- For retrieval use cases, prefer `knowledge_chunks.jsonl` or `retrieval_manifest.json` over direct prompt injection. Use compact memory only when the target agent needs always-on guidance.
- For corpus quality, watch evidence coverage, source diversity, and source concentration. A knowledge pack dominated by one source is weaker even if individual cards look good.
- For human learning, prefer `frontier_learning_report.md`; it should be English, source-attributed, and easier to study than the Agent memory pack.
- Human learning reports must be written as readable learning guides with themes, explanations, and practice questions, not as direct translations of Agent memory entries.
- Human learning reports may be substantially longer than compact memory. Prefer readable sections, comparison tables, glossary, reading path, design checklist, and exercises when the corpus is large.
- Treat similar but more authoritative or clearer sources as rewrite/synthesis candidates instead of automatic duplicates.
- When the user knows an Agent knowledge cutoff date, use memory-pack time filtering to exclude already-known entries.
- For expanded discovery, use a 2025+ broad frontier policy. Include recent papers, repositories, docs, and technical blogs when they teach reusable agent engineering; unknown-date non-authority sources need review.
- For broad runs, prefer automatic search discovery (`akf discover` or `akf run-team --discover`) over manually prepared seed URL files. Treat URL files as optional local supplements or debugging fixtures, not as the default product path.
- Keep topic coverage broad enough for current agent engineering: memory/RAG, knowledge graphs, MCP/protocols, tool use/routing, structured outputs, reasoning/planning, task hardness, stateful runtime, multi-agent handoffs, human review, guardrails, identity/access, observability/evaluation, deployment, cost/latency, coding agents, computer use, multimodal agents, and context engineering.
- Prefer the multi-agent architecture for large-scale runs: discovery/filtering, deep reading, memory synthesis, human learning report writing, and quality evaluation should be separate roles with separate system prompts and handoff artifacts.
- Treat strong research systems as design evidence when they fit the project: staged survey generation, schema-validated extraction, topic clustering, frozen evaluation sets, and explicit next-run feedback loops are preferred over ad hoc one-shot summaries.
- For end-to-end harvesting, prefer `akf run-team` over manually chaining commands unless debugging or recovering one stage. The public UX should be one unified command; users should not need to start each specialist agent separately.
- When implementing search connectors, keep provider-specific code behind a provider-neutral interface, normalize/dedupe URLs before screening, and write durable artifacts (`query_plan`, `search_results`, `candidate_urls`, `discovery_stats`) for reproducibility.
- Large web runs must fail soft per source. Use `--ingestion-timeout` and `--max-markdown-chars` to skip hostile/slow pages and cap token pressure rather than letting one URL block the complete workflow.
- For LLM deep reading, use bounded concurrency (`--llm-extraction-concurrency`) instead of separate user-started agents. Keep concurrency visible in metrics and avoid exceeding provider rate limits.
- When implementing or documenting agent systems, prefer current production patterns found in the harvested corpus: strict tool schemas, sandboxed/truncated tool outputs, entity-isolated memory, memory decay, durable checkpointing, explicit handoff contracts, human review interrupts, and propagated trace context.
- Use `--use-llm-agents` for small high-quality runs when API cost is acceptable; use individual LLM flags when comparing cost/quality by stage.
- After each iteration, remove disposable smoke-test raw artifacts once their knowledge cards, indexes, and iteration notes have been preserved.
- Add focused tests for behavior changes, especially around URL ingestion, Markdown cleanup, and persistence metadata.
- Use English for code comments and concise docstrings. Use comments only when they clarify non-obvious intent.
- Do not introduce broad refactors unless they directly support the current task.

## Self-Upgrade Loop

- After running Agent Knowledge Forge on real sources, inspect the generated artifacts before declaring the work done.
- For new frontier knowledge, prefer a pipeline of discover -> screen -> ingest -> analyze -> integrate memory -> prune obsolete low-value memory.
- When the output is noisy, identify the smallest reusable system improvement rather than manually fixing one artifact.
- Convert every discovered quality issue into a regression test when practical.
- Update `agent.md` only with durable process lessons, project standards, or recurring evaluation criteria.
- Keep external facts in harvested knowledge artifacts, not in project instructions, unless they describe how this project should operate.
- Prefer iterative measurable upgrades: ingest, analyze, inspect, patch, test, and repeat.
- When tuning filters, compare `knowledge_index.md` before and after the change; prioritize fewer high-signal cards over more low-signal cards.
- When LLM screening is available, compare deterministic and LLM-assisted outputs; preserve disagreements as evidence for improving cheaper rules.
- Record meaningful project upgrades in release notes, README updates, or other tracked docs when they affect public behavior.
- Keep any local agent memory focused on current operational context, not long-form user-facing summaries.

## Verification Checklist

- Run the most relevant tests after changes.
- If `uv` is available, prefer `uv run pytest` and `uv run ruff check .`.
- If `uv` is unavailable, use the local virtual environment fallback, such as `.venv/bin/python -m pytest`.
- Report any commands that could not be run and explain the reason.
