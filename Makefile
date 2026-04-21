# AirOne Makefile
# Run `make help` to see all available targets.

PYTHON      := python
PYTEST      := pytest
PIP         := pip
PKG         := airone
TESTS       := tests/
SCRIPTS     := scripts/
RESULTS     := results/

.PHONY: help install dev-install test test-cov benchmark lint format typecheck clean

##@ General

help: ## Show this help menu
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install package (core deps only)
	$(PIP) install -e .

dev-install: ## Install package with all dev dependencies
	$(PIP) install -e ".[dev,ml]"

##@ Testing

test: ## Run full test suite
	$(PYTEST) $(TESTS) -v --tb=short

test-fast: ## Run tests, skip slow ones
	$(PYTEST) $(TESTS) -v -m "not slow" --tb=short

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TESTS) --cov=$(PKG) --cov-report=html --cov-report=term-missing
	@echo "\nCoverage report: htmlcov/index.html"

test-one: ## Run a single test file: make test-one FILE=tests/test_lsh_similarity.py
	$(PYTEST) $(FILE) -v --tb=short

##@ Benchmarks

benchmark: ## Run full deep benchmark suite
	$(PYTHON) $(SCRIPTS)deep_benchmark.py

benchmark-quick: ## Run orchestrated benchmark on corpus only
	$(PYTHON) -c "from airone.benchmarks.runner import BenchmarkRunner; r=BenchmarkRunner(); rep=r.run(['results/corpus/text_medium.txt','results/corpus/json_log_medium.json']); rep.print_table()"

##@ Code Quality

lint: ## Run flake8 linter
	flake8 $(PKG)/ --max-line-length=100 --extend-ignore=E203,W503

format: ## Auto-format with black + isort
	black $(PKG)/ $(TESTS) --line-length=100
	isort $(PKG)/ $(TESTS) --profile=black

typecheck: ## Run mypy type checker
	mypy $(PKG)/ --ignore-missing-imports

##@ Cleanup

clean: ## Remove build artifacts, caches, and coverage data
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-corpus: ## Remove generated benchmark corpus (forces regeneration)
	rm -rf $(RESULTS)corpus/
