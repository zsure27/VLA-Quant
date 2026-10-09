"""B4 protocol gates and fixed 200-episode development analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.extended_peft import sha
from qvla.paired_metrics import comparison
from scripts.audit_extended_peft_eval import audit_case

CASES = ("B3", "B4", "A4", "BF16", "A0")
PHASES = {"micro": (20, 1), "first50": (20, 5), "next50": (25, 5), "last100": (30, 10)}


def require_prior(path, phase):
    record = json.loads(Path(path).read_text())
    if record.get("model_id") != "B4" or record.get("gate") != f"PASS_PROTOCOL_B4_{phase.upper()}":
        raise ValueError("Missing valid prior protocol gate; never continue on success direction")
    # Re-audit every actual trace, rather than trusting a stale PASS string.
    actual = audit(Path(record["root"]), phase)
    if actual != record: raise ValueError("Prior gate inputs or paired results changed")


def audit(root, phase):
    offset, count = PHASES[phase]
    results = {c: audit_case(root / f"eval-{phase}" / c, c, offset, count) for c in CASES}
    base = results["B3"]
    if any(r["keys"] != base["keys"] or r["first"] != base["first"] for r in results.values()):
        raise ValueError("B4 comparison has different manifest keys or first policy-visible inputs")
    rows = [{"task_id": key[0], "init_state_index": key[1],
             **{c: results[c]["success"][i] for c in CASES}} for i, key in enumerate(base["keys"])]
    return {"gate": f"PASS_PROTOCOL_B4_{phase.upper()}", "root": str(root), "model_id": "B4",
            "phase": phase, "classification": "reused development resets; not independent replication",
            "paired_episodes": len(rows), "paired_rows": rows,
            "cases": {c: {"successes": sum(r["success"]), "episodes": len(rows),
                          "query_count": r["query_count"], "sha256": r["sha256"]} for c, r in results.items()},
            "primary": comparison(rows, "B4", "B3"),
            "references": {c: comparison(rows, "B4", c) for c in ("A4", "BF16", "A0")},
            "continue_rule": "protocol only; fixed total 200/config unless resource/quota/error stop"}


def combined(root):
    rows = []
    phases = {}
    for phase in ("first50", "next50", "last100"):
        path = root / f"{phase}-protocol-gate.json"
        require_prior(path, phase)
        phases[phase] = sha(path)
        rows.extend(json.loads(path.read_text())["paired_rows"])
    if len(rows) != 200 or {(r["task_id"], r["init_state_index"]) for r in rows} != {
            (t, r) for t in range(10) for r in range(20, 40)}:
        raise ValueError("Combined cohort must contain exactly 200 unique task/reset pairs")
    primary = comparison(rows, "B4", "B3")
    failures = [{"task_id": r["task_id"], "reset": r["init_state_index"],
                 **{c: r[c] for c in CASES}} for r in rows if not r["B4"] or not r["B3"]]
    return {"gate": "PASS_PROTOCOL_B4_COMBINED200", "model_id": "B4", "paired_rows": rows,
            "phase_sha256": phases, "classification": "development only", "primary": primary,
            "successes": {c: sum(r[c] for r in rows) for c in CASES},
            "references": {c: comparison(rows, "B4", c) for c in ("A4", "BF16", "A0")},
            "failure_pairs": failures,
            "decision": "SUPPORTED_DEVELOPMENT_INCREMENT" if primary["task_cluster_bootstrap_95ci_fraction"][0] > 0
                else "INCONCLUSIVE_INCREMENT" if primary["net"] > 0 else "NO_POSITIVE_INCREMENT",
            "locked": ["Router", "P4", "P5", "offline_final_holdout"],
            "note": "No micro episodes counted; costs and failure-trace diagnosis required in POST-RUN report"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=(*PHASES, "combined"))
    p.add_argument("--output", type=Path)
    p.add_argument("--require-prior", choices=tuple(PHASES))
    a = p.parse_args()
    if a.require_prior:
        require_prior(a.root / f"{a.require_prior}-protocol-gate.json", a.require_prior)
        print(json.dumps({"gate": "PASS_PRIOR_PROTOCOL", "phase": a.require_prior}))
        return
    if a.phase is None or a.output is None: p.error("--phase and --output required for audit")
    if a.output.exists(): raise ValueError("Never overwrite an existing gate")
    record = combined(a.root) if a.phase == "combined" else audit(a.root, a.phase)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate": record["gate"], "primary": record["primary"]}))


if __name__ == "__main__": main()
