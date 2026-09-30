"""Classify the preregistered 035 env-seed smoke from its raw probe output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


POLICY_INPUTS = ("model/pixel_values_bf16_as_float32", "policy/proprio_normalized")


def changed(comparison: dict) -> bool:
    return any(not comparison[name]["same_sha256"] for name in POLICY_INPUTS)


def exact_repeat(comparison: dict) -> bool:
    return all(item["same_sha256"] for item in comparison.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    probe = json.loads((run_dir / "envseed_probe.json").read_text(encoding="utf-8"))
    rows = probe["rows"]
    if sorted(row["task_id"] for row in rows) != list(range(10)):
        raise ValueError("expected exactly the ten Spatial task IDs")
    if any(row["reset_id"] != 5 for row in rows):
        raise ValueError("unexpected official reset index")

    per_task = []
    for row in rows:
        repeat_ok = all(
            exact_repeat(row["within_seed_repeat"][seed][phase])
            for seed in ("0", "1") for phase in ("query0", "after8")
        ) and all(
            exact_repeat(row["within_seed_reset_only_repeat"][seed])
            for seed in ("0", "1")
        )
        valid = all(
            not any(row["task_validity"][seed].values()) for seed in ("0", "1")
        )
        before_fixed = any(
            not item["same_sha256"] for item in row["reset_only_seed_difference"].values()
        )
        query0 = changed(row["query0_seed_difference"])
        after8 = changed(row["after8_seed_difference"])
        per_task.append({
            "task_id": row["task_id"],
            "official_reset_index": row["reset_id"],
            "derived_environment_seeds_differ": row["env_seed_0"] != row["env_seed_1"],
            "same_seed_exact_repeat": repeat_ok,
            "task_valid": valid,
            "reset_before_fixed_state_changed": before_fixed,
            "policy_input_changed_query0": query0,
            "policy_input_changed_after8": after8,
            "query0_bf16_pixel_fraction_changed": row["query0_seed_difference"][POLICY_INPUTS[0]]["fraction_changed"],
            "after8_bf16_pixel_fraction_changed": row["after8_seed_difference"][POLICY_INPUTS[0]]["fraction_changed"],
            "normalized_proprio_changed_query0": not row["query0_seed_difference"][POLICY_INPUTS[1]]["same_sha256"],
        })

    materials_match = (
        (run_dir / "material_sha256_before.txt").read_bytes()
        == (run_dir / "material_sha256_after.txt").read_bytes()
    )
    helper_import_verified = (
        "SEED_ORDER=model_then_environment:PASS"
        in (run_dir / "import_assertion.txt").read_text(encoding="utf-8")
    )
    all_valid = all(row["same_seed_exact_repeat"] and row["task_valid"]
                    and row["derived_environment_seeds_differ"] for row in per_task)
    query0_count = sum(row["policy_input_changed_query0"] for row in per_task)
    after8_count = sum(row["policy_input_changed_after8"] for row in per_task)
    both_count = sum(row["policy_input_changed_query0"] and row["policy_input_changed_after8"]
                     for row in per_task)
    if not materials_match or not helper_import_verified or not all_valid:
        classification = "FAIL_CONTRACT_OR_REPRODUCIBILITY"
    elif both_count >= 8:
        classification = "PASS_CONDITION_SMOKE_ONLY"
    elif any(row["reset_before_fixed_state_changed"] for row in per_task):
        classification = "HOLD_CONDITION_RESET_EFFECT_NOT_POLICY_VISIBLE"
    else:
        classification = "HOLD_CONDITION_NO_RESET_CHANGE"

    pixel_fractions = [row["query0_bf16_pixel_fraction_changed"] for row in per_task]
    summary = {
        "plan_id": "a1-035-envseed-order-fix-smoke-v2-20260930",
        "classification": classification,
        "interpretation_limit": "same official development reset index; this is a policy-input condition smoke, not a C0/C3 result or independent holdout",
        "material_sha256_before_after_identical": materials_match,
        "corrected_helper_import_and_order_verified": helper_import_verified,
        "tasks": len(per_task),
        "same_seed_exact_repeat_tasks": sum(row["same_seed_exact_repeat"] for row in per_task),
        "valid_initial_conditions": sum(row["task_valid"] for row in per_task),
        "reset_before_fixed_state_changed_tasks": sum(row["reset_before_fixed_state_changed"] for row in per_task),
        "policy_input_changed_query0_tasks": query0_count,
        "policy_input_changed_after8_tasks": after8_count,
        "policy_input_changed_at_both_tasks": both_count,
        "query0_bf16_pixel_fraction_changed": {
            "minimum": min(pixel_fractions),
            "mean": mean(pixel_fractions),
            "maximum": max(pixel_fractions),
        },
        "per_task": per_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "classification", "tasks", "same_seed_exact_repeat_tasks",
        "policy_input_changed_at_both_tasks", "query0_bf16_pixel_fraction_changed",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
