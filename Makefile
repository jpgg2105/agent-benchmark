.PHONY: install demo test
install:
	pip install -e ".[dev]"
demo:
	python -m agent_benchmark.cli run --config configs/default.yaml
test:
	pytest -q
