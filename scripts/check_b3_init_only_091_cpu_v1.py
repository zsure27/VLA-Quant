"""Adversarial metadata contracts for the new S0 role; no torch/GPU needed."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qvla.b3_init_only_091_v1 import canonical_sha, validate_metadata, sha
from qvla.extended_peft import FAMILIES


def fixture():
    rows, shapes = [], {}
    for layer in range(32):
        for family in FAMILIES:
            name = f"language_model.model.layers.{layer}.{family}"
            out_size, in_size = ((11008, 4096) if family in ("mlp.gate_proj", "mlp.up_proj")
                                 else (4096, 11008) if family == "mlp.down_proj" else (4096, 4096))
            rows.append({"name": name, "shape": [out_size, in_size], "bits": 2,
                         "group": 64, "rank": 8, "parameters": 8 * (out_size + in_size)})
            shapes[f"{name}.awq_recovery_lora.0.weight"] = [8, in_size]
            shapes[f"{name}.awq_recovery_lora.1.weight"] = [out_size, 8]
    rows.sort(key=lambda row: row["name"])
    initial, teacher = "a" * 64, "b" * 64
    spec = {"rank": 8, "alpha_over_rank": 1, "seed": 7,
            "inherited_cache_sha256": {"initial-adapter.pt": initial, "teacher-targets.pt": teacher}}
    record = {"version": "b3-initialization-v1-20261015", "model_id": "B3",
        "initialization": "response-svd", "steps": 1000, "gate": "PASS_TRAIN_CONTRACT",
        "training_state_space": "policy_normalized_proprio", "zero_output_equal": True,
        "reload_output_equal": True, "smoke_artifact_sha256": "c" * 64,
        "frozen_before_sha256": "d" * 64, "frozen_after_sha256": "d" * 64,
        "parameters": 19988480, "targets": rows, "tensor_shapes": shapes,
        "target_sha256": canonical_sha(rows), "reuse_files": {"initial-adapter.pt": initial},
        "inherited_initial_factor_sha256": initial, "teacher_cache_sha256": teacher}
    training = {"role": "student_state_train", "source_model_id": "A4",
        "state_space": "policy_normalized_proprio", "source_run_sha256": "e" * 64,
        "samples": [{"task_id": t, "init_state_index": r, "query_in_episode": q}
                    for t in range(10) for r in range(4) for q in (0, 1)]}
    return record, spec, training, initial


def main():
    passed = []
    validate_metadata(*fixture())
    passed.append("valid_original_parent_and_pretraining_factor_role")
    cases = {
        "final_adapter_cannot_be_substituted": lambda r, s, t: r["reuse_files"].update({"initial-adapter.pt": "f" * 64}),
        "steps_zero_cannot_impersonate_completed_parent": lambda r, s, t: r.update(steps=0),
        "wrong_proprio_state_space_rejected": lambda r, s, t: r.update(training_state_space="raw_proprio_before_get_action"),
        "changed_frozen_base_rejected": lambda r, s, t: r.update(frozen_after_sha256="f" * 64),
        "changed_rank_rejected": lambda r, s, t: s.update(rank=16),
        "changed_scaling_rejected": lambda r, s, t: s.update(alpha_over_rank=0.125),
        "changed_budget_rejected": lambda r, s, t: r.update(parameters=20000000),
        "missing_target_rejected": lambda r, s, t: r["targets"].pop(),
        "wrong_shape_rejected": lambda r, s, t: r["tensor_shapes"].update({next(iter(r["tensor_shapes"])): [16, 4096]}),
        "dev_role_in_training_rejected": lambda r, s, t: t["samples"][0].update(init_state_index=4),
        "wrong_sampling_policy_rejected": lambda r, s, t: t.update(source_model_id="B5"),
    }
    for label, mutate in cases.items():
        record, spec, training, initial = copy.deepcopy(fixture())
        mutate(record, spec, training)
        try:
            validate_metadata(record, spec, training, initial)
        except (ValueError, KeyError):
            passed.append(label)
        else:
            raise AssertionError(f"Admission unexpectedly accepted {label}")
    result = {"gate": "PASS_B3_INIT_ONLY_CPU_METADATA_CONTRACTS", "contracts": passed,
        "count": len(passed), "gpu_started": False,
        "sources": {name: sha(ROOT / name) for name in
            ("qvla/b3_init_only_091_v1.py", "qvla/run_eval_b3_init_only_091_v1.py",
             "scripts/check_b3_init_only_091_cpu_v1.py")},
        "limits": "Metadata contracts only; real initial-state tensors/model/proprio/reload remain GPU gates"}
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        path = Path(sys.argv[2])
        if path.exists():
            raise ValueError("Do not overwrite a CPU gate")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise ValueError("Usage: check_b3_init_only_091_cpu_v1.py [--output new.json]")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
