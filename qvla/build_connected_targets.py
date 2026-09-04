from __future__ import annotations

import argparse
from pathlib import Path

import torch


EXCLUDED = {
    "vision_backbone.featurizer.blocks.23.attn.proj",
    "vision_backbone.featurizer.blocks.23.attn.qkv",
    "vision_backbone.featurizer.blocks.23.mlp.fc1",
    "vision_backbone.featurizer.blocks.23.mlp.fc2",
    "vision_backbone.fused_featurizer.attn_pool.kv",
    "vision_backbone.fused_featurizer.attn_pool.mlp.fc1",
    "vision_backbone.fused_featurizer.attn_pool.mlp.fc2",
    "vision_backbone.fused_featurizer.attn_pool.proj",
    "vision_backbone.fused_featurizer.attn_pool.q",
    "vision_backbone.fused_featurizer.blocks.26.attn.proj",
    "vision_backbone.fused_featurizer.blocks.26.attn.qkv",
    "vision_backbone.fused_featurizer.blocks.26.mlp.fc1",
    "vision_backbone.fused_featurizer.blocks.26.mlp.fc2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("proxy", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    proxy = torch.load(args.proxy, map_location="cpu")
    targets = [name for name in proxy if name not in EXCLUDED]
    unexpected = EXCLUDED - set(proxy)
    if unexpected:
        raise KeyError(f"Excluded targets absent from proxy: {sorted(unexpected)}")
    if len(proxy) != 435 or len(targets) != 422:
        raise RuntimeError(f"Expected 435/422 targets, got {len(proxy)}/{len(targets)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(targets) + "\n")
    print(f"original={len(proxy)} connected={len(targets)} excluded={len(EXCLUDED)}")
    print(args.output)


if __name__ == "__main__":
    main()
