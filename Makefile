
requirements-all:
	$(MAKE) requirements-txt
	$(MAKE) requirements-txt-dev

requirements-txt:
	uv pip compile pyproject.toml -o requirements.txt

requirements-txt-dev:
	uv pip compile pyproject.toml -o requirements-dev.txt --extra dev