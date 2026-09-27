# Repository Guidelines

## Project Structure & Module Organization

SCOPE is a Python research codebase for embodied navigation evaluation. The root scripts `run_aeqa_evaluation.py` and `run_goatbench_evaluation.py` run the two benchmarks; `llm_match.py` scores A-EQA answers. Shared navigation, mapping, and model code lives in `src/`, including the `src/conceptgraph/` package. Keep benchmark settings in `cfg/`, bundled question and benchmark data in `data/`, unit tests in `tests/`, and project-page files in `docs/`. `assets/` contains README imagery.

## Setup and Evaluation Commands

There is no separate build step. On Linux with CUDA 11.8, create the documented environment with `conda env create -f environment.yml` and `conda activate scope`. The environment references the sibling `../open-eqa` package. Set the HM3D location in the relevant `cfg/eval_*.yaml` file before running an evaluation.

- `python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml` generates A-EQA predictions. Use `--start_ratio 0.0 --end_ratio 0.5` to process a subset.
- `python llm_match.py -cf cfg/openeqa.yaml` scores those predictions; run `python llm_match.py --prepare -cf cfg/openeqa.yaml` first when the answer or judge model changes.
- `python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml` runs GOAT-Bench evaluation.
- `python -m unittest discover -s tests` runs the available unit tests.

## Coding Style & Naming Conventions

Use four-space indentation for Python, `snake_case` for modules and functions, and `PascalCase` for classes. Follow nearby code when editing existing modules. Name new tests `tests/test_*.py` and methods `test_*`. The repository has no committed formatter or linter configuration; avoid unrelated formatting changes.

## Testing Guidelines

Tests currently use the standard-library `unittest` framework. Add focused tests for changed scoring or other deterministic logic, then run the discovery command above. Full benchmark runs require HM3D assets, model weights, GPU support, and a configured VLM endpoint; state which benchmark and configuration you checked when submitting changes. No coverage threshold is configured.

## Commits, Pull Requests & Configuration

Recent commits use brief, descriptive subjects in English or Korean; no fixed prefix convention is evident. Keep commits focused. In pull requests, describe the behavior changed, relevant config or dataset, commands run, and results; link an issue when one exists and include screenshots for `docs/` visual changes. Keep API keys out of commits: the A-EQA pipeline reads VLM settings from `../3dmem/.env`.
