"""只绘制探针实测结果，不生成合成数据，也不输出拟合结论。"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    cases = {}
    for path in sorted(args.root.glob("*/metrics.json")):
        data = json.loads(path.read_text())
        if data:
            cases[path.parent.name] = data
    if not cases:
        raise ValueError("No measured metrics.json found")
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    # 每个用例单独分面，避免几十条曲线挤在无法阅读的图例中。
    for name, data in cases.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, group in zip(axes.flat, ("image_main", "image_wrist", "text", "action_readout")):
            depths, means, lo, hi = [], [], [], []
            for i in range(32):
                key = f"language_model.model.layers.{i}@0/{group}"
                v = [d["features"][key]["relative_mse"] for d in data if key in d["features"]]
                v = [x for x in v if x is not None]
                if v:
                    depths.append(i); means.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
            if depths:
                ax.plot(depths, np.maximum(means, 1e-12), marker=".", label="Mean across samples")
                ax.fill_between(depths, np.maximum(lo, 1e-12), np.maximum(hi, 1e-12), alpha=.15, label="Sample range, not CI")
            ax.set(title=group, xlabel="LLM block index", ylabel="Relative feature MSE (floor 1e-12)", yscale="log")
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(name + " | token-sampled, full-channel features")
        fig.savefig(out / (name + "-llm-depth.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, (branch, call) in zip(axes.flat, (("featurizer", 0), ("featurizer", 1), ("fused_featurizer", 0), ("fused_featurizer", 1))):
            pairs = []
            pattern = re.compile(rf"vision_backbone\.{branch}\.blocks\.(\d+)@{call}/features")
            for key in data[0]["features"]:
                match = pattern.fullmatch(key)
                if match:
                    vals = [d["features"][key]["relative_mse"] for d in data]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        pairs.append((int(match[1]), np.mean(vals)))
            pairs.sort()
            if pairs:
                ax.plot([x for x, y in pairs], np.maximum([y for x, y in pairs], 1e-12), marker=".")
            ax.set(title=f"{branch} / camera call {call}", xlabel="Vision block index",
                   ylabel="Relative feature MSE (floor 1e-12)", yscale="log")
            ax.grid(alpha=.2)
        fig.suptitle(name)
        fig.savefig(out / (name + "-vision-depth.png"), dpi=150)
        plt.close(fig)

    # 动作坐标分别展示，避免混合不可比较的物理单位。
    matrix = np.asarray([np.mean([d["normalized_rmse_per_dim"] for d in data], axis=0) for data in cases.values()])
    fig, ax = plt.subplots(figsize=(9, max(4, len(cases) * .38)), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0)
    ax.set_yticks(range(len(cases)), list(cases))
    ax.set_xticks(range(7), ["x", "y", "z", "rx", "ry", "rz", "gripper"])
    ax.set(xlabel="Normalized action coordinate", ylabel="Fresh-process probe case",
           title="Mean sample-wise RMSE versus BF16 teacher (not success rate)")
    fig.colorbar(im, ax=ax, label="RMSE in normalized action units")
    fig.savefig(out / "action-coordinate-rmse.png", dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
