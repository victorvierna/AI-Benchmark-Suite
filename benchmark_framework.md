# BenchKit Framework Notes

BenchKit is a small, local-first benchmarking harness for LLM tasks. A benchmark is defined by a *suite* (YAML) plus a *dataset* (JSONL). The runner executes each case against one or more models, evaluates outputs, and writes a reproducible results bundle.

## Core Concepts

- **Suite** (`suite.yaml`): Declares the task, prompt template, output schema, and evaluation rules.
- **Dataset** (`cases.jsonl`): One JSON object per line. Each case must have `id` and `input` (object). `expected` is optional but required for most evaluators.
- **Models** (`models.yaml`): Lists provider + model name + parameters. Used to compare multiple models in one run.
- **Pricing** (`pricing/*.yaml`): Optional table for cost estimation (per 1M tokens).
- **Provider**: A thin client that sends requests to an LLM API. v0.1 supports OpenAI Responses.
- **Evaluators**: JSON schema validation + exact field checks; composed via `type: composite`.
- **Results**: `attempts.jsonl` (per-case outputs) and `summary.json` (aggregated metrics). Optional `report.html`.

## Execution Flow

1. Load suite, models, and pricing configs.
2. Load dataset and apply optional filters/limit.
3. For each model, run each case (plus warmup if configured).
4. Evaluate output against suite rules.
5. Record per-attempt data and compute summary metrics.

## Important Files

- `benchkit/cli.py`: CLI commands (validate, run, report, diff, init-suite).
- `benchkit/runner.py`: Execution loop and summary aggregation.
- `benchkit/providers/openai_responses.py`: OpenAI Responses provider.
- `benchkit/evaluators/`: JSON schema + exact field evaluation.
- `benchkit/templates/report.html.j2`: HTML report template.
- `suites/`: Example benchmark suites.
- `results/`: Generated outputs (ignored by git).

## Dataset Format

Each line in `cases.jsonl` must be valid JSON:

```
{"id":"C01","input":{"text":"hello"},"expected":{"label":"greeting"}}
```

`input` is passed into the Jinja2 template as `input.*`.

## Output Controls

Suites can control stored artifacts via `output`:

```
output:
  save_requests: true
  save_raw_responses: true
```

- `save_requests`: store the request payload in attempts.
- `save_raw_responses`: store raw provider responses in attempts.

## Redaction

If the suite defines redaction patterns and redaction is enabled, requests and responses are redacted before being written to disk or included in reports.

## Extending Providers

Add a provider in `benchkit/providers/` implementing `ProviderClient.run()`, then update runner provider selection to support the new provider key in suite configs.
