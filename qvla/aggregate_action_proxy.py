from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


ACTION_KEYS = {0: "prune_0", 2: "asymmetric_2", 4: "asymmetric_4", 8: "asymmetric_8"}


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "__")


def rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--hessian", required=True, type=Path)
    parser.add_argument("--output-proxy", required=True, type=Path)
    parser.add_argument("--output-symmetric", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    targets = [line.strip() for line in args.targets.read_text().splitlines() if line.strip()]
    sample_dirs = sorted(
        path for path in args.input_root.iterdir()
        if path.is_dir() and path.name.startswith("sample-")
    )
    if args.num_samples is not None:
        sample_dirs = sample_dirs[: args.num_samples]
    if len(sample_dirs) < 2:
        raise RuntimeError(f"Need at least two sample directories, got {len(sample_dirs)}")
    num_samples = len(sample_dirs)
    half_samples = num_samples // 2
    all_pairs = list(combinations(range(num_samples), 2))
    if len(all_pairs) > 28:
        pair_indices = np.linspace(0, len(all_pairs) - 1, 28, dtype=int)
        diagnostic_pairs = [all_pairs[index] for index in pair_indices]
    else:
        diagnostic_pairs = all_pairs
    hessian = torch.load(args.hessian, map_location="cpu")
    action_proxy: dict[str, dict[str, torch.Tensor]] = {}
    symmetric_audit: dict[str, dict[str, torch.Tensor]] = {}
    layer_rows = []
    global_action = {bit: [] for bit in ACTION_KEYS}
    global_hessian = {bit: [] for bit in ACTION_KEYS}
    total_channels = 0

    for target in targets:
        samples = []
        for sample_dir in sample_dirs:
            path = sample_dir / f"{safe_name(target)}.pt"
            result = torch.load(path, map_location="cpu")
            if result.get("target") != target or result.get("active") is not True:
                raise RuntimeError(f"Invalid result: {path}")
            samples.append(result["scores"])

        action_proxy[target] = {}
        symmetric_audit[target] = {}
        channels = samples[0]["prune_0"].numel()
        total_channels += channels

        for bit, source_key in ACTION_KEYS.items():
            stack = torch.stack([sample[source_key].float() for sample in samples])
            aggregate = stack.mean(dim=0)
            proxy_key = f"proxy_{bit}"
            expected = hessian[target][proxy_key].float().reshape(-1)
            if aggregate.shape != expected.shape or not torch.isfinite(aggregate).all():
                raise RuntimeError(f"Invalid aggregate for {target} {proxy_key}")
            action_proxy[target][proxy_key] = aggregate
            global_action[bit].append(aggregate)
            global_hessian[bit].append(expected)
            pairwise = [
                rho(stack[i].numpy(), stack[j].numpy())
                for i, j in diagnostic_pairs
            ]
            layer_rows.append({
                "target": target,
                "bit": bit,
                "channels": channels,
                "pairwise_mean": float(np.mean(pairwise)),
                "half_to_full": rho(stack[:half_samples].mean(0).numpy(), aggregate.numpy()),
                "action_hessian_spearman": rho(aggregate.numpy(), expected.numpy()),
            })

        for bit in (2, 4, 8):
            symmetric_audit[target][f"proxy_{bit}"] = torch.stack(
                [sample[f"symmetric_{bit}"].float() for sample in samples]
            ).mean(dim=0)

    if len(targets) != 422 or total_channels != 1_835_680:
        raise RuntimeError(f"Expected 422/1835680, got {len(targets)}/{total_channels}")

    global_rows = []
    for bit in ACTION_KEYS:
        action = torch.cat(global_action[bit]).numpy()
        hp = torch.cat(global_hessian[bit]).numpy()
        relevant = [row for row in layer_rows if row["bit"] == bit]
        global_rows.append({
            "bit": bit,
            "channels": len(action),
            "global_action_hessian_spearman": rho(action, hp),
            "median_layer_action_hessian_spearman": float(np.median([
                row["action_hessian_spearman"] for row in relevant
            ])),
            "median_layer_half_to_full": float(np.median([
                row["half_to_full"] for row in relevant
            ])),
            "layers_half_to_full_at_least_0_9": sum(
                row["half_to_full"] >= 0.9 for row in relevant
            ),
        })

    for path in (args.output_proxy, args.output_symmetric, args.summary_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(action_proxy, args.output_proxy)
    torch.save(symmetric_audit, args.output_symmetric)
    args.summary_json.write_text(json.dumps({
        "targets": len(targets),
        "channels": total_channels,
        "samples": num_samples,
        "half_samples": half_samples,
        "global": global_rows,
        "layers": layer_rows,
    }, indent=2))

    print("targets:", len(targets))
    print("channels:", total_channels)
    for row in global_rows:
        print(row)
    print("saved:", args.output_proxy)
    print("ACTION PROXY AGGREGATION: PASS")


if __name__ == "__main__":
    main()
