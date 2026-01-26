# AGENTS.md

This repo contains **BenchKit**, a lightweight local benchmarking framework for LLM evaluations.

## What it is
- Runs a benchmark suite (YAML) against one or more models.
- Evaluates results with JSON schema + exact field matching.
- Produces `attempts.jsonl`, `summary.json`, and optional `report.html`.

## Key structure
- `benchkit/`: core package and CLI.
- `suites/`: benchmark suites (each has `suite.yaml` + `cases.jsonl`).
- `pricing/`: price tables (optional).
- `results/`: generated outputs (gitignored).
- `tests/`: unit tests.

## Quickstart
```
python -m venv .venv
source .venv/bin/activate
pip install -e .

export OPENAI_API_KEY=...
benchkit validate suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml
benchkit run suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml --runs 1 --report html
```

## Add a new benchmark
1. Scaffold a suite:
   `benchkit init-suite my_suite --base-dir suites`
2. Edit `suites/my_suite/suite.yaml` and `suites/my_suite/cases.jsonl`.
3. Create a models file (or reuse `models.example.yaml`).
4. Run validate + run commands.

## Environment
- Requires Python 3.10+.
- Uses OpenAI Responses API by default.
- `OPENAI_API_KEY` is required for live runs.

## Docs
- Design notes: `benchmark_framework.md`.
