.PHONY: setup smoke test lint clean

setup:
	pip install -e ".[dev]"

setup-wandb:
	pip install -e ".[dev,wandb]"

smoke:
	python -m pytest tests/smoke_test.py -v

test:
	python -m pytest tests/ -v

lint:
	ruff check src tests

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
