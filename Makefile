.PHONY: build shell test verify-gpu gpu

build:
	uv sync --extra vlm

shell:
	uv run bash

test:
	uv run --group dev pytest tests/ -q

verify-gpu:
	uv run --extra vlm python -m src.verify_gpu

# Example: make gpu CMD="python -m src.extraction.run_extraction --source raw"
gpu:
	uv run --extra vlm $(CMD)
