"""Audit the 300-episode fresh B3/B4 pair; keep 200-episode references separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvla.paired_metrics import comparison
from qvla.split_scope import sha, task_inventory
from scripts.audit_extended_peft_eval import audit_case
from scripts.audit_b4_data_scope import verify_receipt, REFERENCES
from scripts.audit_b4_pair_v2 import audit as audit_first_200

CASES = ("B3", "B4")
PHASES = ("micro", "first50", "next50", "last100", "final100")


def audit(root: Path, phase: str) -> dict:
    if phase != "final100":
        result = audit_first_200(root, phase)
        result["gate"] = f"PASS_PROTOCOL_B4_PAIR_V3_{phase.upper()}"
        result["cohort_limit"] = "Archived references cover reset20-39 only; never extend them to reset40-49"
        return result

    receipt = verify_receipt(root / "data-scope-receipt.json")
    inventory = task_inventory()
    results = {case: audit_case(root / "eval-final100" / case, case, 40, 10) for case in CASES}
    base = results["B3"]
    if results["B4"]["keys"] != base["keys"] or results["B4"]["first"] != base["first"]:
        raise ValueError("Final B3/B4 keys or first policy-visible observations differ")
    training = json.loads(Path(receipt["training_manifest"]).read_text(encoding="utf-8"))["samples"]
    train_obs = {row["observation_sha256"] for row in training}
    for case in CASES:
        events = [json.loads(line) for line in (root / "eval-final100" / case / "on-policy-events.jsonl").read_text(encoding="utf-8").splitlines()]
        if train_obs & {row["observation_sha256"] for row in events if row["record_type"] == "query"}:
            raise ValueError("Final evaluation reused a training observation")
    rows = []
    for i, key in enumerate(base["keys"]):
        task, reset, initial_sha = key[:3]
        if initial_sha != inventory[task]["active_state_sha256"][reset]:
            raise ValueError("Final evaluation initial state differs from registered inventory")
        rows.append({"task_id": task, "init_state_index": reset,
                     "B3": base["success"][i], "B4": results["B4"]["success"][i]})
    return {"gate": "PASS_PROTOCOL_B4_PAIR_V3_FINAL100", "phase": phase,
            "classification": "same-task reused development reset40-49; no concurrent reference",
            "paired_rows": rows, "paired_episodes": len(rows),
            "data_scope_sha256": sha(root / "data-scope-receipt.json"),
            "cases": {case: {"successes": sum(result["success"]), "episodes": len(rows),
                             "sha256": result["sha256"]} for case, result in results.items()},
            "primary": comparison(rows, "B4", "B3"),
            "archival_references": None,
            "continue_rule": "protocol only, never success direction"}


def require_prior(root: Path, phase: str) -> None:
    saved = json.loads((root / f"{phase}-protocol-gate.json").read_text(encoding="utf-8"))
    if saved != audit(root, phase):
        raise ValueError("Prior gate differs from actual inputs or outputs")


def combined(root: Path) -> dict:
    rows = []
    for phase in PHASES[1:]:  # Exclude overlapping micro episodes.
        require_prior(root, phase)
        rows.extend(json.loads((root / f"{phase}-protocol-gate.json").read_text(encoding="utf-8"))["paired_rows"])
    expected = {(task, reset) for task in range(10) for reset in range(20, 50)}
    if len(rows) != 300 or {(r["task_id"], r["init_state_index"]) for r in rows} != expected:
        raise ValueError("Require 300 unique paired episodes on reset20-49")
    historical = [r for r in rows if r["init_state_index"] < 40]
    if len(historical) != 200 or any(not all(ref in r for ref in REFERENCES) for r in historical):
        raise ValueError("Archived references must be present on exactly reset20-39")
    if any(any(ref in r for ref in REFERENCES) for r in rows if r["init_state_index"] >= 40):
        raise ValueError("Archived reference was incorrectly extended to reset40-49")
    primary = comparison(rows, "B4", "B3")
    return {"gate": "PASS_PROTOCOL_B4_PAIR_V3_COMBINED300",
            "classification": "same-task development; not unseen-task or independent reproduction",
            "paired_rows": rows, "primary": primary,
            "successes_300": {case: sum(r[case] for r in rows) for case in CASES},
            "archival_reference_20_39": {
                "episodes": 200,
                "successes": {ref: sum(r[ref] for r in historical) for ref in REFERENCES},
                "B4_comparisons": {ref: comparison(historical, "B4", ref) for ref in REFERENCES},
                "limit": "Matched historical inputs only; no 300-episode BF16/A4/A0 rate"},
            "failure_pairs": [r for r in rows if not r["B3"] or not r["B4"]],
            "decision": "SUPPORTED_DEVELOPMENT_INCREMENT" if primary["task_cluster_bootstrap_95ci_fraction"][0] > 0
                        else "INCONCLUSIVE_INCREMENT" if primary["net"] > 0 else "NO_POSITIVE_INCREMENT",
            "locked": ["Router", "P4", "P5", "offline_final_holdout"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=(*PHASES, "combined"))
    p.add_argument("--output", type=Path)
    p.add_argument("--require-prior", choices=PHASES)
    a = p.parse_args()
    if a.require_prior:
        require_prior(a.root, a.require_prior)
        print("PASS_PRIOR_PROTOCOL")
        return
    if not a.phase or not a.output:
        p.error("--phase and --output required")
    if a.output.exists():
        raise ValueError("Never overwrite a gate")
    result = combined(a.root) if a.phase == "combined" else audit(a.root, a.phase)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"gate": result["gate"], "primary": result["primary"]}))


if __name__ == "__main__":
    main()
