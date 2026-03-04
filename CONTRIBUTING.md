# Contributing to Tailr

Thanks for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/Tailr.git
cd Tailr
uv sync --all-extras
```

## Development Workflow

1. Create a branch: `git checkout -b my-feature`
2. Make your changes
3. Run checks:
   ```bash
   uv run ruff check src/
   uv run mypy src/
   uv run pytest
   ```
4. Commit and push
5. Open a pull request

## Code Style

- Formatted and linted with [Ruff](https://docs.astral.sh/ruff/)
- Type-checked with [mypy](https://mypy-lang.org/) (strict mode)
- Line length: 100 characters

## Reporting Issues

Open an issue with:
- What you expected
- What happened instead
- Steps to reproduce

