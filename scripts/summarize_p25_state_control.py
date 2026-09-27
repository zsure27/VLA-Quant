"""Compare no-LoRA and LoRA on identical student-visited observations."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.original / "bf16-same-observation/same-observation-comparisons.jsonl").read_text().splitlines()]
    baseline = json.loads((args.control / "exact12l-no-lora/metrics.json").read_text())
    if len(rows) != len(baseline) or len(rows) != 140:
        raise RuntimeError("Expected 140 paired student observations")
    paired = []
    teacher_max = 0.0
    for index, (row, metric) in enumerate(zip(rows, baseline)):
        name = f"sample-{index:06d}"
        if metric["sample"] != name + ".npz" or row["file"] != f"policy-observations/query-{index:06d}.npz":
            raise RuntimeError(f"Order mismatch at {index}")
        with np.load(args.original / "bf16-same-observation" / row["teacher_response_file"], allow_pickle=False) as source:
            teacher = source["teacher_action"].astype(np.float32)
        probe = torch.load(args.control / "bf16-probe" / f"{name}.pt", map_location="cpu", weights_only=True)["raw"].numpy()
        teacher_max = max(teacher_max, float(np.max(np.abs(teacher - probe))))
        paired.append({"index": index, "episode_serial": row["episode_serial"],
                       "lora_mse": float(row["chunk_mse"]),
                       "no_lora_mse": float(metric["raw_action"]["mse"]),
                       "delta_lora_minus_no_lora": float(row["chunk_mse"] - metric["raw_action"]["mse"]),
                       "lora_gripper_disagreement": float(row["gripper_disagreement"]),
                       "no_lora_gripper_disagreement": float(metric["raw_gripper_disagreement"])})
    if teacher_max != 0.0:
        raise RuntimeError(f"BF16 teacher mismatch; comparison invalid: max_abs={teacher_max}")
    per_episode = {}
    for row in paired:
        per_episode.setdefault(row["episode_serial"], []).append(row)
    output = {"schema_version": "1.0", "paired_queries": len(paired), "episodes": len(per_episode),
              "bf16_teacher_max_abs_difference": teacher_max,
              "lora_mean_mse": statistics.mean(row["lora_mse"] for row in paired),
              "no_lora_mean_mse": statistics.mean(row["no_lora_mse"] for row in paired),
              "mean_paired_delta_lora_minus_no_lora": statistics.mean(row["delta_lora_minus_no_lora"] for row in paired),
              "fraction_queries_lora_better": statistics.mean(row["delta_lora_minus_no_lora"] < 0 for row in paired),
              "fraction_episodes_lora_better": statistics.mean(statistics.mean(r["delta_lora_minus_no_lora"] for r in group) < 0 for group in per_episode.values()),
              "lora_gripper_disagreement": statistics.mean(row["lora_gripper_disagreement"] for row in paired),
              "no_lora_gripper_disagreement": statistics.mean(row["no_lora_gripper_disagreement"] for row in paired),
              "per_episode": [{"episode_serial": key, "queries": len(group),
                               "mean_delta": statistics.mean(r["delta_lora_minus_no_lora"] for r in group)}
                              for key, group in sorted(per_episode.items())],
              "classification": "student-visited mechanism diagnosis; development resets; not a closed-loop causal effect"}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "per_episode"}))


if __name__ == "__main__":
    main()
