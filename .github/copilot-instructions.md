# Copilot Instructions for Tailr

## Project overview

Tailr is a Python CLI tool that evaluates a candidate's CV against a job
description using an LLM and produces an evidence-based fit report.

## Tech stack

| Layer        | Technology                                               |
|--------------|----------------------------------------------------------|
| Language     | Python 3.13 (strict mypy, ruff linting)                 |
| Packaging    | `uv` with `pyproject.toml` (`uv_build` backend)         |
| CLI          | Typer + Rich                                             |
| LLM          | LangChain (langchain-openai, langchain-google-genai)     |
| Models       | Pydantic v2 (structured output from LLM)                 |
| HTTP         | httpx (provider model listing)                           |
| Testing      | pytest                                                   |

## Repository layout

```
src/tailr/
├── __init__.py          # Package root, exports __version__
├── analyzer.py          # FitAnalysis model, FitAnalyzer class, LLM prompts
├── providers.py         # Provider enum, model-listing HTTP clients
├── report.py            # Markdown report generation and file saving
├── scrubber.py          # PII scrubber — regex-based removal of personal info
├── cli/
│   ├── main.py          # Top-level Typer app, registers sub-commands
│   └── fit.py           # `tailr fit` command — orchestrates analysis flow
tests/
├── conftest.py          # Shared fixtures (make_analysis, make_analyzer, json_response)
├── test_analyzer.py     # Tests for analyzer error handling + provider listing
├── test_report.py       # Tests for report generation and file output
└── test_scrubber.py     # Tests for PII scrubbing (email, phone, URLs, name, address)
```

## Architecture & data flow

1. **CLI** (`cli/fit.py`): parses args, loads CV + job files.
2. **Scrubber** (`scrubber.py`): strips PII from CV text before LLM call
   (emails, phones, LinkedIn/GitHub URLs, generic URLs, addresses, candidate
   name). Bypass with `--no-scrub`.
3. **Analyzer** (`analyzer.py`): builds a LangChain prompt + structured output
   chain, invokes the LLM, returns a `FitAnalysis` Pydantic model.
4. **Report** (`report.py`): renders `FitAnalysis` to markdown, saves to disk.

## Coding conventions

- **Type hints everywhere** — `mypy --strict` must pass.
- **`from __future__ import annotations`** at the top of every module.
- **Ruff** for linting and formatting — line length 100, rules: E, F, I, N, W, UP.
- **Docstrings** on every public function/class (Google-ish style, one-liner or
  multi-line).
- **No wildcard imports** — always import specific names.
- Use `match`/`case` for provider dispatch (Python 3.13 pattern matching).
- Pydantic `BaseModel` for structured data; `Field(description=...)` on every field.
- Constants use `_UPPER_SNAKE` (module-private leading underscore).
- Tests use `pytest` with class-based grouping and parametrize where appropriate.
- Fixture factories (e.g. `make_analyzer(**overrides)`) in `conftest.py`.

## PII scrubbing rules (`scrubber.py`)

Order matters — specific patterns before generic:

1. Email → `[EMAIL]`
2. LinkedIn URL → `[LINKEDIN]`
3. GitHub URL → `[GITHUB]`
4. Generic URL → `[URL]`
5. Street address → `[ADDRESS]`
6. Phone number → `[PHONE]`
7. Candidate name (first name-like line) → `[NAME]`

Name detection heuristic: first non-blank, non-header line that is 2–5
alphabetic words and under 60 chars. Replaced globally throughout the document.

## Adding a new feature — checklist

1. Create or edit source files under `src/tailr/`.
2. Add or update tests under `tests/`.
3. Update `README.md` if the change is user-facing.
4. Run **`make all`** and confirm it passes (lint → format → typecheck → test → build).

## Validation command

After any implementation work, always run:

```bash
make all
```

This runs `lint`, `format`, `typecheck`, `test`, and `build` in sequence.
**All steps must pass before the work is considered complete.**

