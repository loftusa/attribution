# Attribution: Data and Model Experiments for Code LLMs

## Overview
This repository explores attribution and evaluation for code-focused language models. It includes:
- Utilities for dataset handling and prompt generation (CruxEval utilities).
- Benchmarks on HumanEval/EvalPlus and CruxEval before/after tuning.
- Tuning workflows on APPS and evaluations across OLMo checkpoints.
- Analysis notebooks and scripts for comparing pass rates, plotting progress, and sanity checks.

Core package lives in `src/attribution/` (utils, benchmark, vis, tune). Experiments, figures, and helper scripts live in `experiments/`.

## Setup
- Python 3.12+. Dependencies managed with `uv` (see `pyproject.toml`).
- Install: `uv sync`
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Some experiments require GPU and tokens (e.g., `HF_TOKEN`).

## Quick Start
- Demo (toy flow): `uv run python src/attribution/main.py`
- Benchmark a model (HumanEvalPlus via bigcode-eval harness):
  - Edit model name in `src/attribution/benchmark.py` or call `benchmark(model_name)`.
  - Run: `uv run python src/attribution/benchmark.py`
- Fine-tune on APPS (HF Transformers Trainer):
  - `uv run python src/attribution/tune/apps_train.py --help`

## Experiments Summary (experiments/)
- HumanEval/EvalPlus
  - `00_human_eval.py`: Evaluate pass@k with EvalPlus helpers.
- Training & Multi-benchmark Analysis
  - `02_train_single_benchmark.py`: Train on a single benchmark.
  - `02_01_single_benchmark_results.py`, `03_multiple_benchmark_result.py`, `04_assess_multiple_benchmarks.py`:
    Aggregate, compare, and score multiple runs across datasets.
  - `05_compare_pass_rates.py`: Compare pass rates across settings.
- Function Composition & Activation Probing
  - `06_function_composition.py`: Composition tests; behavior under chaining.
  - `11_test_activation_norm.py`: Activation norm sanity checks.
- CruxEval Suite
  - `07_evaluate_cruxeval.py` + `07_*plots*.py`: Evaluate and plot CruxEval results.
- OLMo and Checkpoints
  - `assess_olmo_models.sh`, `07_evaluate_olmo_models.sh`: Evaluate OLMo variants.
  - `10_download_checkpoints.{sh,py}`, `10_plot_base.py`, `10_test_revisions.py`,
    `10_evaluate_{base,instruct}.sh`, `10_grab_branches.py`: Manage/evaluate checkpoints.
  - Progress artifacts: `accuracy_by_stage.html`, `stage{1,2}_progress.html`.
- Utilities
  - `benchmark.sh`, `finetune.sh`, `multipleE_bigcode.sh`, `get_multiplek_results.sh`.

## Notes
- The bigcode-eval harness is configured as an editable local source in `pyproject.toml`.
  Update `[tool.uv.sources]` if your path differs.
- For a lightweight sanity check without network/GPU: `uv run python src/test_format.py`.
