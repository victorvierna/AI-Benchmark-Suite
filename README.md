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
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

You can also store them in a `.env` file at the repo root:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Quickstart

```bash
# Validate a suite
benchkit validate suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml

# Run a suite
benchkit run suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml --runs 1
```

WhatsApp importance classifier benchmark across OpenAI, Gemini, and Anthropic:

```bash
benchkit validate suites/whatsapp_importance_classifier/suite.yaml --models models.classifier.yaml --pricing pricing/cloud.yaml
benchkit run suites/whatsapp_importance_classifier/suite.yaml --models models.classifier.yaml --pricing pricing/cloud.yaml --runs 1 --report html
```

Use `models.cloud.yaml` instead of `models.classifier.yaml` to run the broader current model list.

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
- Use `OPENAI_API_KEY` for OpenAI, `GEMINI_API_KEY` for Gemini, and `ANTHROPIC_API_KEY` for Anthropic.
- For LM Studio (local OpenAI-compatible server), set `provider: lmstudio` in your suite and optionally `LMSTUDIO_BASE_URL` (default: `http://localhost:1234/v1`).
