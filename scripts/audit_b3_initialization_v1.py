"""Gate fixed-A4 training comparability and paired Z/S rollout protocols."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvla.extended_peft import sha, canonical_sha


def audit_training(root, steps):
    import torch
    records = {v: json.loads((root / v / f"steps-{steps}" / "artifact.json").read_text())
               for v in ("standard-zero", "response-svd")}
    z, s = records.values()
    expected_gate = "PASS_SMOKE10" if steps == 10 else "PASS_TRAIN_CONTRACT"
    for variant, record in records.items():
        if record.get("gate") != expected_gate or record.get("steps") != steps or record.get("initialization") != variant:
            raise ValueError("Wrong training candidate/gate")
        if not record.get("zero_output_equal") or not record.get("reload_output_equal"):
            raise ValueError("Numerical zero/reload gate incomplete")
        if record["frozen_before_sha256"] != record["frozen_after_sha256"]:
            raise ValueError("Frozen base changed")
        if record["parameters"] != 19988480:
            raise ValueError("Not the fixed B3 budget")
    for field in ("checkpoint_identity", "targets", "target_sha256", "training_manifest_sha256",
                  "source_sha256", "spec_sha256", "sample_sequence_sha256", "frozen_before_sha256",
                  "teacher_cache_sha256"):
        if z[field] != s[field]:
            raise ValueError(f"Uncontrolled difference between Z/S: {field}")
    roots = [root / v / f"steps-{steps}" for v in records]
    teachers = [torch.load(p / "teacher-targets.pt", map_location="cpu", weights_only=True) for p in roots]
    if set(teachers[0]) != set(teachers[1]) or any(not torch.equal(teachers[0][k], teachers[1][k]) for k in teachers[0]):
        raise ValueError("Teacher labels differ")
    inputs = [json.loads((p / "teacher-input-hashes.json").read_text()) for p in roots]
    if inputs[0] != inputs[1]:
        raise ValueError("Processed teacher inputs differ")
    state = torch.load(roots[0] / "initial-adapter.pt", map_location="cpu", weights_only=True)
    for key, value in state.items():
        zero = torch.count_nonzero(value).item() == 0
        if zero != (".awq_recovery_lora.1." in key):
            raise ValueError("Z requires random nonzero A and exactly zero B")
    return {"gate": "PASS_B3_INITIALIZATION_SMOKE_PAIR" if steps == 10 else "PASS_B3_INITIALIZATION_TRAIN_PAIR",
            "steps": steps, "records": {v: sha(p / "artifact.json") for v, p in zip(records, roots)},
            "parameters_each": 19988480, "sample_sequence_sha256": z["sample_sequence_sha256"],
            "teacher_cache_sha256": z["teacher_cache_sha256"],
            "limits": "Training and structural admission only; no closed-loop benefit conclusion"}


def audit_rollout(root, offset, count):
    from scripts.audit_extended_peft_eval import audit_case
    from qvla.paired_metrics import comparison
    variants = ("standard-zero", "response-svd")
    records = {}
    for variant in variants:
        directory = root / variant
        command = json.loads((directory / "invocation.json").read_text())["command"]
        if command[command.index("--b3-initialization") + 1] != variant:
            raise ValueError("Actual rollout candidate differs from registered variant")
        if command[command.index("--weight-bits") + 1] != "2" or command[command.index("--activation-bits") + 1] != "16":
            raise ValueError("Actual rollout precision changed")
        records[variant] = audit_case(directory, "B3", offset, count)
        log = next(directory.glob("EVAL-*.txt")).read_text()
        if re.search(r"Episode error:|Traceback \(most recent call last\)", log):
            raise ValueError("Program error in completed rollout")
        from qvla.reproducibility import episode_seeds
        for key in records[variant]["keys"]:
            model_seed, env_seed = episode_seeds(0, 1, "libero_spatial", key[0], key[1])
            if key[3:] != (env_seed, model_seed, "paired"):
                raise ValueError("Actual derived episode seeds differ from registered paired protocol")
    z, s = records.values()
    if z["keys"] != s["keys"] or z["first"] != s["first"]:
        raise ValueError("Z/S do not share exact reset/seed/first-observation pairing")
    rows = [{"task_id": k[0], "init_state_index": k[1], **{v: records[v]["success"][i] for v in variants}}
            for i, k in enumerate(z["keys"])]
    return {"gate": "PASS_B3_INITIALIZATION_PAIRED_PROTOCOL", "paired_episodes": len(rows),
            "paired_rows": rows, "comparison_S_minus_Z": comparison(rows, "response-svd", "standard-zero"),
            "cases": {v: {"successes": sum(r["success"]), "queries": r["query_count"], "sha256": r["sha256"]}
                      for v, r in records.items()},
            "limits": "Reused same-task development; continuation decided by protocol/resources, not score direction"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--steps", type=int, choices=(10, 1000))
    p.add_argument("--offset", type=int)
    p.add_argument("--count", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Never overwrite a gate")
    if a.steps is None and (a.offset is None or a.count is None):
        p.error("Rollout audit requires offset/count")
    result = audit_training(a.root, a.steps) if a.steps else audit_rollout(a.root, a.offset, a.count)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"gate": result["gate"], "output": str(a.output)}))


if __name__ == "__main__":
    main()
