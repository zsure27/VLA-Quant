from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from .common import read_json, write_json
from .plan import compile_plan


def main():
    parser = argparse.ArgumentParser(description="VLA suite × quantization × finetuning experiment infrastructure")
    parser.add_argument("--plugin", action="append", default=[], help="Explicit trusted module registering new capabilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List registered capabilities without loading a model")
    plan = sub.add_parser("plan", help="Compile a new immutable round plan; never starts GPU")
    plan.add_argument("config", type=Path)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Execute only on a current, explicitly authorized Linux server")
    run.add_argument("plan", type=Path)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--round-id", required=True)
    run.add_argument("--quota", type=Path, required=True)
    run.add_argument("--execute", action="store_true", required=True)
    status = sub.add_parser("status", help="Read a single small status snapshot")
    status.add_argument("run_dir", type=Path)
    metrics = sub.add_parser("aggregate")
    metrics.add_argument("episodes", type=Path)
    metrics.add_argument("--output", type=Path, required=True)
    pair = sub.add_parser("compare")
    pair.add_argument("episodes", type=Path)
    pair.add_argument("--baseline", required=True)
    pair.add_argument("--candidate", required=True)
    pair.add_argument("--output", type=Path, required=True)
    server = sub.add_parser("ssh", help="Print SSH argv using local environment, no login or boot")
    server.add_argument("server_id")
    server.add_argument("--table", type=Path, default=Path("configs/infra/servers.json"))
    report = sub.add_parser("report", help="Refresh one task document after editing its SUMMARY.md")
    report.add_argument("task_id")
    legacy = sub.add_parser("legacy-links", help="Optional server-only symlinks for frozen historical scripts")
    legacy.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for module in args.plugin:
        importlib.import_module(module)
    repo = Path(__file__).resolve().parents[2]
    if args.command == "list":
        from .registry import SUITES, QUANTIZERS, STRATEGIES
        print(json.dumps({"suites": SUITES, "quantization": QUANTIZERS, "finetuning": STRATEGIES}, ensure_ascii=False, indent=2))
    elif args.command == "plan":
        if args.output.exists():
            parser.error("Plan output already exists; use a new round directory")
        value = compile_plan(read_json(args.config), repo, args.run_dir)
        write_json(args.output, value)
        print(json.dumps({"plan_sha256": value["plan_sha256"], "stages": [x["id"] for x in value["stages"]]}))
    elif args.command == "run":
        from .runner import execute
        execute(read_json(args.plan), args.authorization, args.round_id, args.quota)
    elif args.command == "status":
        print(json.dumps(read_json(args.run_dir / "status.json"), ensure_ascii=False, indent=2))
    elif args.command == "aggregate":
        from .metrics import export
        export(args.episodes, args.output)
    elif args.command == "compare":
        from .metrics import paired
        write_json(args.output, paired(read_json(args.episodes), args.baseline, args.candidate))
    elif args.command == "ssh":
        from .lifecycle import resolve_server, ssh_argv
        print(json.dumps(ssh_argv(resolve_server(args.table, args.server_id))))
    elif args.command == "report":
        from .report import render
        render(repo / "Ex_Reports", read_json(repo / "Ex_Reports/_tasks" / args.task_id / "task.json"))
    elif args.command == "legacy-links":
        for top in ("reports", "results"):
            for target in (repo / "Ex_Reports/_legacy" / top).iterdir():
                if not target.is_dir():
                    continue
                link = repo / top / target.name
                if link.exists() or link.is_symlink():
                    if link.is_symlink() and link.resolve() == target.resolve():
                        continue
                    raise ValueError("Existing legacy location; do not overwrite: " + str(link))
                print(str(link) + " -> " + str(target))
                if args.apply:
                    link.symlink_to(target, target_is_directory=True)


if __name__ == "__main__":
    main()
