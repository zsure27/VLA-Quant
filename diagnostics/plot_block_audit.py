"""Show matched-input AWQ block errors from an existing, completed audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.audit / "complete.json").read_text())
    if complete.get("status") != "COMPLETE" or not complete.get("teacher_restored"):
        raise ValueError("Block audit did not complete or teacher restoration failed")
    manifest = json.loads((args.audit / "manifest.json").read_text())
    records = json.loads((args.audit / "metrics.json").read_text())
    layers = manifest["layers"]
    variants = list(manifest["variants"])
    samples = sorted(manifest["samples"])
    if len(records) != len(layers) * len(variants) * len(samples) * 2:
        raise ValueError("Missing or duplicate records")
    lookup = {(r["layer"], r["variant"], r["sample"], r["context"]): r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate record key")
    expected = {(l, v, s, c) for l in layers for v in variants for s in samples
                for c in ("teacher", "w2_upstream")}
    if set(lookup) != expected:
        raise ValueError("Unexpected block audit keys")
    contexts = ("teacher", "w2_upstream")
    values = {}
    for context in contexts:
        matrix = np.array([[np.median([lookup[l, v, s, context]
                            ["same_input_error"]["action_tokens"]["relative_mse"]
                            for s in samples]) for v in variants] for l in layers])
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("Non-finite or negative block errors")
        values[context] = matrix
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), sharey=True)
    for ax, context in zip(axes, contexts):
        arr = values[context]
        visual = np.log10(np.maximum(arr, 1e-10))
        im = ax.imshow(visual, aspect="auto", vmin=-10, vmax=0, cmap="magma")
        ax.set(xticks=range(len(variants)), xticklabels=variants,
               yticks=range(len(layers)), yticklabels=layers,
               ylabel="LLM layer", title=f"Matched input: {context}")
        ax.tick_params(axis="x", rotation=45)
        for row in range(len(layers)):
            for col in range(len(variants)):
                ax.text(col, row, f"{arr[row, col]:.1e}", ha="center", va="center",
                        fontsize=7, color="white" if visual[row, col] < -3 else "black")
    fig.colorbar(im, ax=axes, label="log10 median relative MSE of action-token hidden states")
    fig.suptitle("AWQ local block perturbation; 3 held-out samples, 5 selected layers\n"
                 "Input drift is held fixed within each panel; neither panel estimates closed-loop success")
    fig.subplots_adjust(top=0.76, bottom=0.28)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "block-error-heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    result = {"layers": layers, "variants": variants, "samples": samples,
              "median_relative_mse_action_tokens": {k: v.tolist() for k, v in values.items()},
              "interpretation": "Selected local blocks, held-out samples, old profile and checkpoint; no intervention has been validated in a full closed-loop W2 policy."}
    (args.output / "block-error-summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
