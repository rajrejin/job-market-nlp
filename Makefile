check:
	ruff check . && ruff format --check . && mypy && pytest
