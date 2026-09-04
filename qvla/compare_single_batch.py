from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", required=True, type=Path)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()

    reference = torch.load(args.single, map_location="cpu")
    target = reference["target"]
    candidate_path = args.batch_dir / f"{safe_name(target)}.pt"
    candidate = torch.load(candidate_path, map_location="cpu")
    if candidate["target"] != target:
        raise RuntimeError(f"Target mismatch: {candidate['target']} != {target}")

    print("target:", target)
    passed_all = True
    for key, expected in sorted(reference["scores"].items()):
        actual = candidate["scores"][key].float().reshape(-1)
        expected = expected.float().reshape(-1)
        difference = (actual - expected).abs()
        denominator = expected.abs().sum().clamp_min(1e-30)
        normalized_l1 = float(difference.sum() / denominator)
        actual64 = actual.double()
        expected64 = expected.double()
        norm_product = actual64.norm() * expected64.norm()
        cosine = float((actual64 @ expected64) / norm_product) if norm_product > 0 else 1.0
        sum_ratio = float(actual64.sum() / expected64.sum()) if expected64.sum() != 0 else 1.0
        rank = float(spearmanr(actual.numpy(), expected.numpy()).statistic)
        k = max(1, round(expected.numel() * 0.01))
        expected_top = set(np.argpartition(expected.numpy(), -k)[-k:])
        actual_top = set(np.argpartition(actual.numpy(), -k)[-k:])
        top1 = len(expected_top & actual_top) / k
        allclose = torch.allclose(actual, expected, rtol=args.rtol, atol=args.atol)
        ranking_equivalent = rank >= 0.999 and cosine >= 0.9999 and normalized_l1 <= 0.01 and top1 >= 0.98
        passed = bool(allclose or ranking_equivalent)
        passed_all &= passed
        print(
            key,
            "max_abs=", float(difference.max()),
            "normalized_l1=", normalized_l1,
            "sum_ratio=", sum_ratio,
            "cosine=", cosine,
            "spearman=", rank,
            "top1=", top1,
            "allclose=", allclose,
            "pass=", passed,
        )
    if passed_all:
        print("SINGLE/BATCH EQUIVALENCE: PASS")
    elif args.diagnostic_only:
        print("SINGLE/BATCH EQUIVALENCE: DIAGNOSTIC ONLY")
    else:
        raise AssertionError("Batch mismatch under value and ranking criteria")


if __name__ == "__main__":
    main()
