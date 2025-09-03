# Repository Guidelines

## Project Structure & Modules
- Source: `src/attribution/` (core package: utils, benchmark, vis, tune/).
- Experiments: `experiments/` (scripts, notebooks, and helper shells for benchmarks and plots).
- Data: `data/` (inputs, datasets, or caches you add locally).
- Results: `results/` (generated outputs, metrics, figures).
- Config/build: `pyproject.toml` (hatchling build; dependencies managed via `uv`).

## Build, Test, and Dev Commands
- Install env: `uv sync` — create/resolve environment from `pyproject.toml` and `uv.lock`.
- Lint: `uv run ruff check .` — static checks; add `--fix` to auto‑fix.
- Format: `uv run ruff format .` — format codebase.
- Quick check: `uv run python src/test_format.py` — sanity run for prompt formatting.
- Run scripts:
  - Benchmark: `uv run python src/attribution/benchmark.py`
  - APPS fine‑tune: `uv run python src/attribution/tune/apps_train.py --help`

Notes: `src/attribution/main.py` is a demo; run as a script path (not as a module) due to local imports.

## Coding Style & Naming
- Language: Python 3.12+; 4‑space indentation; UTF‑8.
- Style: Keep functions/variables `snake_case`, classes `CamelCase`, modules `lower_snake_case.py`.
- Imports: Prefer absolute package imports within `attribution` (e.g., `from attribution.utils import ...`).
- Lint/format: Enforce with Ruff (check + format) before pushing.

## Testing Guidelines
- Lightweight checks are script‑based (see `src/test_format.py`, `experiments/` scripts).
- Property tests: Hypothesis is available; prefer adding new tests under `tests/` with `pytest` if introduced later.
- Include minimal, fast checks that run without GPUs or network when possible.

## Commit & Pull Requests
- Commits: Use concise, imperative subjects (e.g., "add cruxeval loader", "fix ruff issues").
- Scope: Separate experimental changes from library code. Place one concern per commit.
- PRs must include:
  - Purpose and summary of changes; link any related issues.
  - How to reproduce (commands, dataset pointers); attach small sample outputs in `results/` if relevant.
  - Confirmation that `uv sync` succeeds and `ruff check .` passes.

## Security & Configuration Tips
- Avoid committing large artifacts or credentials. Respect `.gitignore`.
- Some scripts require tokens or GPUs (e.g., Hugging Face models, OpenAI). Export needed env vars (`HF_TOKEN`, `OPENAI_API_KEY`) locally.
