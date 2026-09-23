"""Validate the P0 mixed-group plan against two complete AWQ profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnostics.awq_interventions import attention_no_clip_primary_g64_plan


def load(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["entries"], payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("group64", type=Path)
    args = parser.parse_args()
    scales, targets, removed = attention_no_clip_primary_g64_plan(
        *load(args.base), load(args.group64)
    )
    summary = {
        "block_scales": len(scales),
        "targets": len(targets),
        "removed_language_clips": len(removed),
        "group_counts": {
            str(group): sum(value[2] == group for value in targets.values())
            for group in sorted({value[2] for value in targets.values()})
        },
        "language_targets": sum(name.startswith("language_model.") for name in targets),
        "dino_targets": sum(name.startswith("vision_backbone.featurizer.") for name in targets),
        "siglip_targets": sum(name.startswith("vision_backbone.fused_featurizer.") for name in targets),
    }
    expected = {
        "block_scales": 32,
        "targets": 422,
        "removed_language_clips": 64,
        "group_counts": {"64": 317, "128": 105},
        "language_targets": 224,
        "dino_targets": 93,
        "siglip_targets": 105,
    }
    if summary != expected:
        raise RuntimeError(f"unexpected P0 plan: {summary}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
