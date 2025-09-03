# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research codebase focused on attribution analysis and model evaluation for language models, particularly code generation models. The project evaluates models on benchmarks like HumanEval+, CruxEval, and APPS, with capabilities for fine-tuning and causal tracing analysis.

## Common Development Commands

### Package Management
- Install dependencies: `uv sync`
- Add new package: `uv add <package_name>`
- Run scripts: `uv run <script>.py`

### Model Evaluation
- Evaluate on HumanEval+: `bash experiments/benchmark.sh`
- Evaluate on CruxEval: `uv run experiments/07_evaluate_cruxeval.py --model_name <model_name>`
- Fine-tune on APPS: `bash experiments/finetune.sh`

### Code Quality
- Format code: `uv run ruff format`
- Lint code: `uv run ruff check`

## Architecture Overview

### Core Components

**src/attribution/**
- `main.py`: Entry point demonstrating model loading, generation, benchmarking, and tuning pipeline
- `benchmark.py`: Wrapper for bigcode-evaluation-harness to evaluate models on code benchmarks
- `utils.py`: Core utilities including CruxEvalUtil for dataset handling, causal tracing functions, and template formatting
- `vis.py`: Visualization utilities for analysis results
- `tune/`: Fine-tuning module for APPS dataset with Hugging Face Trainer

**experiments/**
- Numbered experiment scripts (00-12) for various analyses and evaluations
- Shell scripts for running evaluations and fine-tuning
- Jupyter notebooks for exploratory analysis

### Key Utilities

**CruxEvalUtil Class** (`utils.py:26-84`)
- Loads and processes CruxEval dataset for code execution prediction
- Provides formatted prompts with function/input/output triplets
- Core utility for attribution experiments

**Causal Tracing** (`utils.py:143-204`)
- Implements causal tracing for model interpretability
- Analyzes how different tokens contribute to model predictions
- Uses nnsight for model intervention and analysis

**Template Formatting** (`utils.py:223-327`)
- Handles tokenization and indexing for intervention experiments
- Supports both "last" and "all" token retrieval modes
- Critical for precise token-level interventions

### External Dependencies

**bigcode-evaluation-harness**: Located at `/home/lofty/code_llm/bigcode-evaluation-harness`
- Editable installation for code benchmark evaluation
- Used via `accelerate launch` commands in shell scripts

## Model Support

The codebase supports various model families:
- OLMo models (base, instruct, DPO variants)
- Code-specific models (CodeLlama, StarCoder)
- General language models with code capabilities

## Environment Requirements

- GPU support required for model inference and training
- HUGGING_FACE_HUB_TOKEN environment variable for model access
- Weights & Biases integration for experiment tracking
- Python 3.12+ with uv package manager

## Data Organization

**data/**: Contains evaluation results, plots, and processed datasets
**results/**: Model-specific evaluation outputs organized by provider/model name
**experiments/**: Experiment scripts with clear numerical ordering for workflow

## Experiment Workflow

1. Model loading and basic generation testing (`main.py`)
2. Baseline evaluation on benchmarks (`benchmark.py`, shell scripts)
3. Fine-tuning on APPS dataset (`tune/` module)
4. Causal analysis and attribution studies (numbered experiments)
5. Result visualization and analysis (`vis.py`, plotting scripts)