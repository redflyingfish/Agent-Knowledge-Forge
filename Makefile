PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
AKF ?= .venv/bin/akf
RUFF ?= .venv/bin/ruff
PYTEST ?= .venv/bin/pytest
YEAR ?= 2026
OUT ?= data/broad-run
RUN_DIR ?= $(OUT)

.PHONY: help venv install install-dev playwright env-check llm-check lint test check discover topic-discovery run smoke mcp-server clean

help:
	@echo "Agent Knowledge Forge commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv          Create .venv"
	@echo "  make install       Install runtime requirements"
	@echo "  make install-dev   Install runtime + editable dev package"
	@echo "  make playwright    Install Playwright Chromium dependencies"
	@echo "  make env-check     Check local .env exists"
	@echo "  make llm-check     Validate LLM config without spending tokens"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run ruff"
	@echo "  make test          Run pytest"
	@echo "  make check         Run lint + tests"
	@echo ""
	@echo "Pipeline:"
	@echo "  make discover      Run search discovery only"
	@echo "  make topic-discovery  Mine candidate emerging topics from RUN_DIR"
	@echo "  make run           Run the full LLM-assisted pipeline"
	@echo "  make smoke         Run a tiny explicit-URL smoke test"
	@echo "  make mcp-server    Serve RUN_DIR as a read-only MCP server"
	@echo ""
	@echo "Variables:"
	@echo "  YEAR=2026 OUT=data/broad-run RUN_DIR=data/broad-run"

venv:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"

playwright:
	$(PYTHON) -m playwright install --with-deps chromium

env-check:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@echo ".env exists"

llm-check: env-check
	$(AKF) llm-check --stage screening

lint:
	$(RUFF) check src/agent_knowledge_harvester tests

test:
	$(PYTHON) -m pytest -q

check: lint test

discover: env-check
	$(AKF) discover \
		--year $(YEAR) \
		--max-queries 40 \
		--results-per-query 5 \
		--out $(OUT)/00_discovery

topic-discovery:
	$(AKF) topic-discovery \
		--in-dir $(RUN_DIR)/02_ingested \
		--search-report $(RUN_DIR)/00_discovery/search_results.json \
		--out $(RUN_DIR)/07_topic_discovery

run: env-check
	$(AKF) run-team \
		--discover \
		--discovery-year $(YEAR) \
		--use-llm-agents \
		--report-language both \
		--llm-max-candidates 25 \
		--max-selected-urls 80 \
		--include-review \
		--concurrency 4 \
		--llm-extraction-concurrency 2 \
		--ingestion-timeout 45 \
		--max-markdown-chars 120000 \
		--out $(OUT)

smoke: env-check
	$(AKF) run-team \
		--url https://modelcontextprotocol.io/specification/2025-06-18 \
		--use-llm-agents \
		--out data/smoke-run

mcp-server:
	$(AKF) mcp-server --run-dir $(RUN_DIR)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
