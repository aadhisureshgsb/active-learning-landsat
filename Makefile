.PHONY: install dev run test clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

run:
	python -m active_learning

test:
	pytest -q

clean:
	rm -rf data results/*.png __pycache__ */__pycache__ .pytest_cache *.egg-info
