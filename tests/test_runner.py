import json
import os

from benchkit.config import load_models, load_pricing, load_suite
from benchkit.runner import run_suite
from benchkit.types import LLMRequest, LLMResponse, Usage


class FakeProvider:
    def run(self, request: LLMRequest, timeout_s: int = 60) -> LLMResponse:
        # Always return a JSON that matches expected label=ok
        return LLMResponse(
            text='{ "label": "ok" }',
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            raw={"ok": True},
        )


def test_run_suite(tmp_path):
    suite_path = "tests/fixtures/simple_suite.yaml"
    suite = load_suite(suite_path)
    models = load_models("tests/fixtures/models.yaml")
    pricing = load_pricing("tests/fixtures/pricing.yaml")

    out_dir = tmp_path / "results"
    results_dir = run_suite(
        suite=suite,
        suite_path=suite_path,
        models=models,
        pricing=pricing,
        provider_override=FakeProvider(),
        runs=1,
        limit=2,
        out_dir=str(out_dir),
    )

    assert os.path.isdir(results_dir)
    summary_path = os.path.join(results_dir, "summary.json")
    attempts_path = os.path.join(results_dir, "attempts.jsonl")
    assert os.path.exists(summary_path)
    assert os.path.exists(attempts_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["models"][0]["passed"] == 2
