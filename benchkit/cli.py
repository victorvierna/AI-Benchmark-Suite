from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .config import load_models, load_pricing, load_suite, ValidationError
from .dataset import DatasetError, filter_cases, load_cases
from .doctor import run_doctor
from .report import print_console_summary, render_html_report, load_json
from .diff import diff_runs, print_diff
from .runner import run_suite


def _resolve_path(path: str) -> str:
    return os.path.abspath(path)

def _parse_duration(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s.endswith("ms"):
        return int(float(s[:-2]) / 1000.0)
    if s.endswith("s"):
        return int(float(s[:-1]))
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    return int(float(s))

def cmd_validate(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
    except (ValidationError, Exception) as e:
        print(f"Suite config error: {e}")
        return 2

    if args.models:
        try:
            load_models(args.models)
        except (ValidationError, Exception) as e:
            print(f"Models config error: {e}")
            return 2

    if args.pricing:
        try:
            load_pricing(args.pricing)
        except (ValidationError, Exception) as e:
            print(f"Pricing config error: {e}")
            return 2

    try:
        dataset_path = os.path.join(os.path.dirname(args.suite), suite.dataset.path)
        cases = load_cases(dataset_path)
        cases = filter_cases(cases, args.filter, args.limit)
        if not cases:
            print("No cases after filters/limit")
            return 2
    except DatasetError as e:
        print(f"Dataset error: {e}")
        return 2

    print("Validation OK")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    return run_doctor(args.suite, args.models, args.pricing, ping=args.ping)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
    except (ValidationError, Exception) as e:
        print(f"Suite config error: {e}")
        return 2

    if not args.models:
        print("--models is required")
        return 2

    try:
        models = load_models(args.models)
    except (ValidationError, Exception) as e:
        print(f"Models config error: {e}")
        return 2

    pricing = None
    if args.pricing:
        try:
            pricing = load_pricing(args.pricing)
        except (ValidationError, Exception) as e:
            print(f"Pricing config error: {e}")
            return 2

    # validate dataset early
    try:
        dataset_path = os.path.join(os.path.dirname(args.suite), suite.dataset.path)
        load_cases(dataset_path)
    except DatasetError as e:
        print(f"Dataset error: {e}")
        return 2

    out_dir = args.out
    results_dir = run_suite(
        suite=suite,
        suite_path=args.suite,
        models=models,
        pricing=pricing,
        runs=args.runs,
        limit=args.limit,
        filters=args.filter,
        warmup=args.warmup,
        max_cost_usd=args.max_cost_usd,
        max_requests=args.max_requests,
        max_time_s=_parse_duration(args.max_time),
        out_dir=out_dir,
        redact=args.redact,
    )

    summary = load_json(os.path.join(results_dir, "summary.json"))
    print_console_summary(summary)

    if args.report == "html":
        render_html_report(results_dir, suite.redact.patterns if suite.redact else None)
        print(f"HTML report generated in {results_dir}")

    print(f"Results saved in {results_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.format == "html":
        render_html_report(args.results_dir)
        print(f"HTML report generated in {args.results_dir}")
        return 0
    print("Only html format supported in v0.1")
    return 2


def cmd_diff(args: argparse.Namespace) -> int:
    diff = diff_runs(args.run_a, args.run_b)
    print_diff(diff)
    if args.out:
        import json

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=2)
    return 0


def cmd_init_suite(args: argparse.Namespace) -> int:
    suites_dir = args.base_dir
    os.makedirs(suites_dir, exist_ok=True)
    suite_dir = os.path.join(suites_dir, args.suite_id)
    os.makedirs(suite_dir, exist_ok=True)

    suite_yaml = os.path.join(suite_dir, "suite.yaml")
    cases_jsonl = os.path.join(suite_dir, "cases.jsonl")
    readme = os.path.join(suite_dir, "README.md")

    if not os.path.exists(suite_yaml):
        with open(suite_yaml, "w", encoding="utf-8") as f:
            f.write(
                """id: {id}
version: 1
provider: openai

request:
  type: openai_responses
  system: |
    You are a helpful classifier.
  user_template: |
    Classify this text: \"{{ input.text }}\"
  response_format:
    type: json_schema
    name: result
    strict: true
    schema:
      type: object
      properties:
        label: { type: string }
      required: [label]
      additionalProperties: false
  params:
    temperature: 0

dataset:
  path: cases.jsonl

evaluation:
  mode: binary
  evaluator:
    type: composite
    all:
      - type: json_schema
      - type: exact_fields
        fields:
          - path: $.label
            expected_from: expected.label
""".format(id=args.suite_id)
            )

    if not os.path.exists(cases_jsonl):
        with open(cases_jsonl, "w", encoding="utf-8") as f:
            f.write('{"id":"C01","input":{"text":"hello"},"expected":{"label":"greeting"}}\n')

    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(f"# {args.suite_id}\n\nDescribe your suite here.\n")

    print(f"Suite scaffold created in {suite_dir}")
    return 0


def cmd_list_suites(args: argparse.Namespace) -> int:
    base = args.base_dir
    if not os.path.isdir(base):
        print("No suites directory")
        return 2
    for name in sorted(os.listdir(base)):
        if os.path.isdir(os.path.join(base, name)):
            print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchkit")
    sub = p.add_subparsers(dest="command")

    v = sub.add_parser("validate")
    v.add_argument("suite")
    v.add_argument("--models")
    v.add_argument("--pricing")
    v.add_argument("--limit", type=int)
    v.add_argument("--filter", action="append")
    v.set_defaults(func=cmd_validate)

    d = sub.add_parser("doctor")
    d.add_argument("suite")
    d.add_argument("--models")
    d.add_argument("--pricing")
    d.add_argument("--ping", action="store_true")
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("run")
    r.add_argument("suite")
    r.add_argument("--models", required=True)
    r.add_argument("--pricing")
    r.add_argument("--runs", type=int, default=1)
    r.add_argument("--limit", type=int)
    r.add_argument("--filter", action="append")
    r.add_argument("--warmup", type=int, default=0)
    r.add_argument("--max-cost-usd", type=float)
    r.add_argument("--max-requests", type=int)
    r.add_argument("--max-time")
    r.add_argument("--out")
    r.add_argument("--report", choices=["html"], default=None)
    r.add_argument("--redact", action="store_true")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report")
    rep.add_argument("results_dir")
    rep.add_argument("--format", choices=["html"], default="html")
    rep.set_defaults(func=cmd_report)

    df = sub.add_parser("diff")
    df.add_argument("run_a")
    df.add_argument("run_b")
    df.add_argument("--out")
    df.set_defaults(func=cmd_diff)

    init = sub.add_parser("init-suite")
    init.add_argument("suite_id")
    init.add_argument("--base-dir", default="suites")
    init.set_defaults(func=cmd_init_suite)

    ls = sub.add_parser("list-suites")
    ls.add_argument("--base-dir", default="suites")
    ls.set_defaults(func=cmd_list_suites)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
