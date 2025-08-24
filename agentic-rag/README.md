# Agentic RAG Demo

This repository contains a minimal Python project demonstrating an agentic
retrieval-augmented generation (RAG) pipeline using [Chroma DB](https://www.trychroma.com/).
It includes utilities for web scraping, optional integration with external AI
systems, and an example `pytest` test.

## Structure

- `src/agentic_rag/` – core package
- `tests/` – test suite
- `pyproject.toml` – project metadata and dependencies

## Running tests

Install dependencies (optional extras may require system packages):

```bash
pip install -e .[full]
```

Run the tests:

```bash
pytest
```

Tests that require unavailable packages will be automatically skipped.
