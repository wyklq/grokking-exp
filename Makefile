PYTHON ?= python3
RUFF ?= ruff

.PHONY: setup smoke test lint clean

setup:
	$(PYTHON) -m pip install -e ".[dev]"

setup-wandb:
	$(PYTHON) -m pip install -e ".[dev,wandb]"

smoke:
	$(PYTHON) -m pytest tests/smoke_test.py -v

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(RUFF) check src tests scripts

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
