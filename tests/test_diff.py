import json
import os

from benchkit.diff import diff_runs


def _write_run(dir_path, pass_value: bool):
    os.makedirs(dir_path, exist_ok=True)
    summary = {
        "suite_id": "simple",
        "suite_version": 1,
        "started_at": "x",
        "finished_at": "y",
        "git": {},
        "config": {},
        "models": [
            {
                "provider": "openai",
                "name": "gpt-test",
                "label": "test",
                "pass_rate": 1.0 if pass_value else 0.0,
                "passed": 1 if pass_value else 0,
                "total": 1,
                "latency_ms_p50": 10,
                "latency_ms_p95": 10,
                "latency_ms_avg": 10,
                "cost_usd_total": 0.0,
                "cost_usd_avg": 0.0,
                "tokens_total": 1,
            }
        ],
    }
    with open(os.path.join(dir_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)

    attempt = {
        "model": {"provider": "openai", "name": "gpt-test"},
        "case_id": "C1",
        "eval": {"pass": pass_value},
    }
    with open(os.path.join(dir_path, "attempts.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(attempt) + "\n")


def test_diff_runs(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_run(str(run_a), pass_value=True)
    _write_run(str(run_b), pass_value=False)

    diff = diff_runs(str(run_a), str(run_b))
    assert diff["case_changes"][0]["from"] is True
    assert diff["case_changes"][0]["to"] is False
