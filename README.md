# BenchKit (local)

Minimal LLM benchmarking framework used in this repo. The design is documented in `benchmark_framework.md` at the repo root.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For live runs, set your API key:

```bash
export OPENAI_API_KEY=...
```

## Quickstart

```bash
# Validate a suite
benchkit validate suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml

# Run a suite
benchkit run suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml --runs 1
```

Scaffold a new suite:

```bash
benchkit init-suite my_suite --base-dir suites
```

## Files

- `benchkit/`: core package
- `suites/`: benchmark suites
- `pricing/`: price tables
- `results/`: generated outputs (ignored by git)

## Notes

- No network calls are made unless you run `benchkit run` or `benchkit doctor --ping`.
- Use `OPENAI_API_KEY` for OpenAI provider.
