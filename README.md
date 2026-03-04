# Tailr

LLM-assisted fit analysis CLI. Evaluates a CV against a job description
and returns evidence-based readiness signals, interview talk tracks, and gap strategies.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) or pip
- API key for [OpenAI](https://platform.openai.com/api-keys)
  or [Google Gemini](https://aistudio.google.com/apikey)

## Install

```bash
git clone https://github.com/YOUR_USERNAME/Tailr.git
cd Tailr
uv sync
```

## Setup

```bash
export LLM_API_KEY="your-api-key"
```

Or pass it with `--api-key`.

## Usage

```bash
tailr fit --cv path/to/cv.md --job path/to/job.txt
```

### Options

| Flag            | Short | Description                 | Default                            |
|-----------------|-------|-----------------------------|------------------------------------|
| `--cv`          | `-c`  | CV file path                | *required*                         |
| `--job`         | `-j`  | Job description file path   | *required*                         |
| `--provider`    | `-p`  | `gemini` or `openai`        | `gemini`                           |
| `--model`       | `-m`  | Model name                  | `gemini-2.5-flash` / `gpt-4o-mini` |
| `--output`      | `-o`  | Output directory            | `output/`                          |
| `--api-key`     | `-k`  | API key (overrides env var) | —                                  |
| `--temperature` | `-t`  | Temperature (0.0–2.0)       | `0.7`                              |
| `--max-tokens`  |       | Max output tokens           | `4000`                             |

### Examples

```bash
tailr fit -c cv.md -j job.txt
tailr fit -c cv.md -j job.txt -p openai
tailr fit -c cv.md -j job.txt -m gpt-4o -o results/
```

## Output

Markdown report saved to the output directory (e.g. `fit-backend-engineer.md`).

Contents: fit score (0–100%), verdict (`STRONG FIT`/`POSSIBLE FIT`/`WEAK FIT`),
proof points, risk flags (credibility concerns, not missing keywords), growth
opportunities, interview focus areas, and gap strategies.

## Development

```bash
uv sync --all-extras    # install with dev deps
uv run ruff check src/  # lint
uv run mypy src/        # type check
uv run pytest           # test
```

## License

MIT — see [LICENSE](LICENSE).
