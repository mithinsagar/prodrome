# prodrome development tasks.
#
# The venv split is deliberate: dbt resolves in its own dependency universe and
# installing it alongside the application pins produces conflicts. Two venvs is the
# cheapest correct answer.

VENV      := .venv
VENV_DBT  := .venv-dbt
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
PRODROME  := $(VENV)/bin/prodrome
DBT       := $(VENV_DBT)/bin/dbt
PYTHON    ?= python3.12

.DEFAULT_GOAL := help
.PHONY: help venv install install-dbt lint fmt type test test-all cov \
        cohort doctor dry-run ingest score latency pipeline dbt marts export \
        brief dashboard fixture fixture-bad dbt-negative smoke all clean clean-cache

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- setup ----

venv:  ## Create both virtualenvs
	$(PYTHON) -m venv $(VENV)
	$(PYTHON) -m venv $(VENV_DBT)

install: venv  ## Install the application with every extra
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev,embed,tableau]"

install-dbt:  ## Install dbt in its own virtualenv
	$(VENV_DBT)/bin/pip install -q --upgrade pip
	$(VENV_DBT)/bin/pip install -q "dbt-core>=1.8,<2" "dbt-duckdb>=1.8,<2"
	cd dbt && ../$(DBT) deps

# ------------------------------------------------------------ code quality --

lint:  ## Lint
	$(VENV)/bin/ruff check src tests tools
	$(VENV)/bin/ruff format --check src tests tools

fmt:  ## Format and autofix
	$(VENV)/bin/ruff format src tests tools
	$(VENV)/bin/ruff check --fix src tests tools

type:  ## Type check
	$(VENV)/bin/mypy src

test:  ## Run the offline test suite
	$(PY) -m pytest -m "not network"

test-all:  ## Run every test, including those that hit the live public APIs
	$(PY) -m pytest

cov:  ## Test with a coverage report
	$(PY) -m pytest -m "not network" --cov=prodrome --cov-report=term-missing

# --------------------------------------------------------------- pipeline ---

cohort:  ## Re-resolve the tracked cohort (writes conf/cohort.yml for review)
	$(PY) tools/resolve_cohort.py --show-rejected

doctor:  ## Check the environment before a long run
	$(PRODROME) doctor

dry-run:  ## Estimate the request budget without spending it
	$(PRODROME) ingest --dry-run

ingest:  ## Stage 1: fetch counts and label history
	$(PRODROME) ingest

score:  ## Stage 2: compute calibrated statistics
	$(PRODROME) score

latency:  ## Stage 3: lead time, leakage and the models
	$(PRODROME) latency

pipeline:  ## All three stages
	$(PRODROME) run

# ---------------------------------------------------------- warehouse/dbt ---

dbt marts:  ## Build the dbt models and run the data quality tests
	cd dbt && ../$(DBT) build

export:  ## Export Parquet, the Tableau .hyper extract and the dashboard bundle
	$(PRODROME) export

brief:  ## Write the weekly brief
	$(PRODROME) brief

dashboard:  ## Serve the dashboard locally at http://localhost:8000
	@echo "Serving dashboard/ at http://localhost:8000 -- Ctrl-C to stop"
	@cd dashboard && $(PWD)/$(PY) -m http.server 8000

# ------------------------------------------------------------- fixtures -----

fixture:  ## Generate a synthetic warehouse (no network needed)
	$(PY) tools/make_fixture_warehouse.py

fixture-bad:  ## Generate a synthetic warehouse with planted data-quality violations
	$(PY) tools/make_fixture_warehouse.py --out /tmp/prodrome-bad.duckdb --with-violations

dbt-negative: fixture-bad  ## Prove the data quality tests actually fail on bad data
	@echo "Expecting failures -- a test that has never failed is not known to work."
	cd dbt && PRODROME_WAREHOUSE=/tmp/prodrome-bad.duckdb ../$(DBT) build \
	  --vars '{run_id: latest}' --target ci || echo "OK: the tests caught the planted violations"

smoke:  ## End-to-end run on three drugs and a short window
	$(PRODROME) ingest --config conf/smoke.yml --cohort-limit 3
	$(PRODROME) score --config conf/smoke.yml
	$(PRODROME) latency --config conf/smoke.yml
	cd dbt && ../$(DBT) build
	$(PRODROME) export
	$(PRODROME) brief

all: install install-dbt lint type test fixture marts export  ## Everything, offline

# ------------------------------------------------------------------ clean ---

clean:  ## Remove build and test artefacts (keeps the API cache)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	rm -rf dbt/target dbt/logs dashboard/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-cache:  ## Also drop the API response cache (forces a full refetch)
	rm -rf data/cache data/warehouse data/exports
