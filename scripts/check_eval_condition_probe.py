"""Fail-closed check of a small, evaluator-order condition probe.

This checks observable differences and repeatability only. Source-call-order,
data overlap, one-variable design, and the first policy rollout remain separate
mandatory gates; a PASS here is never authorization for a long evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICY_VISIBLE = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
EVAL_ORDER = "env.seed(derived_env_seed), seed_all(model_seed), env.reset(), set_init_state"


def check(probe, min_tasks, min_changed_tasks, min_pixel_fraction, min_proprio_max_abs):
    reasons = []
    fixed = probe.get("fixed", {})
    if fixed.get("seed_call_order") != EVAL_ORDER:
        reasons.append("probe did not record the frozen evaluator seed call order")
    rows = probe.get("rows")
    if not isinstance(rows, list) or len(rows) < min_tasks:
        return ["fewer task rows than preregistered minimum"] + reasons, 0
    if len({row.get("task_id") for row in rows}) != len(rows):
        reasons.append("duplicate task IDs")
    changed_tasks = 0
    for row in rows:
        task = row.get("task_id")
        if not row.get("initial_state_sha256"):
            reasons.append("task {} missing initial-state hash".format(task))
        repeat = row.get("within_seed_repeat")
        if not isinstance(repeat, dict):
            reasons.append("task {} missing same-condition repeats".format(task))
        else:
            for seed in ("0", "1"):
                for stage in ("query0", "after_fixed_8"):
                    values = repeat.get(seed, {}).get(stage, {})
                    if not values or any(
                        key not in values or not values[key].get("same_sha256", False)
                        for key in POLICY_VISIBLE
                    ):
                        reasons.append("task {} seed {} {} is not repeatable".format(task, seed, stage))
        stage_changed = []
        for stage in ("query0_difference", "after_fixed_8_difference"):
            values = row.get(stage, {})
            if any(key not in values for key in POLICY_VISIBLE):
                reasons.append("task {} {} missing policy-visible observation".format(task, stage))
                stage_changed.append(False)
                continue
            stage_changed.append(any(
                not values[key].get("same_sha256", True)
                and values[key].get("fraction_changed", 0.0) >= min_pixel_fraction
                for key in ("agentview_image", "robot0_eye_in_hand_image")
            ) or any(
                not values[key].get("same_sha256", True)
                and values[key].get("max_abs", 0.0) >= min_proprio_max_abs
                for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
            ))
        if all(stage_changed):
            changed_tasks += 1
    if changed_tasks < min_changed_tasks:
        reasons.append("only {}/{} tasks changed at both policy query points".format(changed_tasks, len(rows)))
    return reasons, changed_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--min-tasks", type=int, default=10)
    parser.add_argument("--min-changed-tasks", type=int, default=10)
    parser.add_argument("--min-pixel-fraction", type=float, default=0.001)
    parser.add_argument("--min-proprio-max-abs", type=float, default=1e-5)
    args = parser.parse_args()
    if args.min_tasks < 1 or not 1 <= args.min_changed_tasks <= args.min_tasks:
        parser.error("require 1 <= min-changed-tasks <= min-tasks")
    if not 0 <= args.min_pixel_fraction <= 1:
        parser.error("min-pixel-fraction must be within [0, 1]")
    if args.min_proprio_max_abs <= 0:
        parser.error("min-proprio-max-abs must be positive")
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    reasons, changed = check(probe, args.min_tasks, args.min_changed_tasks,
                             args.min_pixel_fraction, args.min_proprio_max_abs)
    print(json.dumps({
        "gate": "FAIL" if reasons else "PASS_CONDITION_SMOKE_ONLY",
        "changed_tasks": changed,
        "reasons": reasons,
        "next_gate": "source/order, data overlap, one-variable contract, then paired policy micro-rollout",
    }, ensure_ascii=False))
    raise SystemExit(1 if reasons else 0)


if __name__ == "__main__":
    main()
