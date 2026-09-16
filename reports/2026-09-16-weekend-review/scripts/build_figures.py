"""Build report figures and CSV sidecars from committed evidence, without GPU.

Run: python reports/2026-09-16-weekend-review/scripts/build_figures.py
Requires numpy and matplotlib. No source PDF, credential, server, or network access.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parents[1]
DATA = REPORT / "data"
FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180,
                     "svg.hashsalt": "vla-weekend-review-20260916"})
SOURCE_FILES = set()


def read_json(path):
    path = REPO / path
    SOURCE_FILES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    path = REPO / path
    SOURCE_FILES.add(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (DATA / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def save(fig, name, note):
    fig.text(.02, .025, note, fontsize=8, va="bottom")
    fig.subplots_adjust(bottom=.23, top=.82, left=.10, right=.97, wspace=.32)
    if name == "04_vision_offline":
        fig.axes[1].set_yticklabels([])
        fig.subplots_adjust(left=.23, wspace=.24)
    fig.savefig(FIG / (name + ".png"))
    fig.savefig(FIG / (name + ".svg"), metadata={"Date": None})
    plt.close(fig)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    d = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, mid - d), min(1, mid + d)


gate = read_json("results/awq-baseline-gate-20260916/analysis/summary.json")
screen = read_json("results/awq-visual-diagnostics-20260915/summary.json")
clip = read_json("results/awq-w2-language-clip10-20260916/paired_w2_clip_comparison.json")
records = []
for method, k in [("BF16", gate["all_50"]["bf16_successes"]),
                  ("AWQ W4A16", gate["all_50"]["w4_successes"])]:
    lo, hi = wilson(k, 50)
    records.append(dict(cohort="current_paired_50", method=method, successes=k,
                        episodes=50, rate=k/50, wilson_low=lo, wilson_high=hi,
                        evidence_class="committed_run", repeated_states=10))
for label, key in [("BF16", "BF16 same loader"), ("W4", "W4 repeat + trace"),
                   ("Full W2 candidate", "AWQ W2 tuned"),
                   ("Language W2 no clip", "W2 language only"),
                   ("Vision W2", "W2 vision only")]:
    row = screen[key]
    lo, hi = wilson(row["successes"], 10)
    records.append(dict(cohort="current_screen_10", method=label,
                        successes=row["successes"], episodes=10,
                        rate=row["successes"]/10, wilson_low=lo, wilson_high=hi,
                        evidence_class="committed_run", repeated_states=10))
lo, hi = wilson(clip["clipped_successes"], 10)
records.append(dict(cohort="current_screen_10", method="Language W2 clipped",
                    successes=clip["clipped_successes"], episodes=10,
                    rate=clip["clipped_successes"]/10, wilson_low=lo,
                    wilson_high=hi, evidence_class="committed_run", repeated_states=10))
write_csv("closed_loop.csv", records)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, cohort in zip(axes, ["current_paired_50", "current_screen_10"]):
    rows = [r for r in records if r["cohort"] == cohort]
    y = np.arange(len(rows))
    p = np.array([r["rate"] for r in rows])
    ax.barh(y, p, color=["#526778", "#259e91", "#b14845", "#d69335", "#577dcc", "#b14845"][:len(rows)])
    ax.errorbar(p, y, xerr=np.maximum(0, np.array([[r["rate"]-r["wilson_low"] for r in rows],
                 [r["wilson_high"]-r["rate"] for r in rows]])), fmt="none", color="black", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["method"] for r in rows])
    for i, r in enumerate(rows):
        ax.text(min(r["rate"]+.025, .88), i-.16,
                "{}/{}".format(r["successes"], r["episodes"]), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Closed-loop task success rate")
    ax.set_title("Matched 50-state W4 gate" if cohort.endswith("50") else "10-state screening: attribution only")
fig.suptitle("OpenVLA-OFT Spatial: W4 runs; W2 language remains the bottleneck", fontsize=14)
save(fig, "01_closed_loop", "BF16-carried fake quantization; no recovery training. Bars: observed success; whiskers: descriptive Wilson 95% intervals.\nFixed task strata and repeated states limit generalization. The 10-state runs overlap the 50-state gate; do not pool them.")

paired = read_csv("results/awq-baseline-gate-20260916/analysis/paired_episodes.csv")
write_csv("paired_50.csv", paired)
matrix = np.zeros((10, 5), dtype=int)
for r in paired:
    b, w = r["bf16_success"].lower() == "true", r["w4_success"].lower() == "true"
    matrix[int(r["task_id"]), int(r["init_state_index"])] = 3 if b and w else 1 if b else 2 if w else 0
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(9, 6.6))
ax.imshow(matrix, cmap=ListedColormap(["#cfd4da", "#d38c39", "#668dc4", "#339e8c"]), vmin=0, vmax=3, aspect="auto")
for i in range(10):
    for j in range(5):
        ax.text(j, i, ["F / F", "P / F", "F / P", "P / P"][matrix[i, j]],
                ha="center", va="center", fontsize=8, color="white" if matrix[i, j] else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(["Init {}{}".format(i, " (seen)" if i == 0 else "") for i in range(5)])
ax.set_yticks(range(10)); ax.set_yticklabels(["Task {}".format(i) for i in range(10)])
ax.set_title("BF16 / AWQ W4 paired outcomes: 47/50 each", pad=14)
save(fig, "02_paired_w4", "Cell order: BF16 / W4; P = pass, F = fail. One BF16-only success, one W4-only success; both discordances are in task 9.\nPaired difference = 0 percentage points; conditional within-task bootstrap interval [-6, +6] pp. Equivalence is not established.")

e2e = read_json("results/awq-block-audit-20260916/e2e-interventions-old-summary.json")
rows = []
samples = []
for key, label in [("baseline", "W2 clipped"), ("no-clip-all", "W2 no extra clip"),
                   ("no-clip-attention", "Attention no clip"), ("no-clip-mlp", "MLP no clip"),
                   ("w4-blocks-10-12", "W4 blocks 10-12; rest clipped")]:
    r = e2e[key]
    rows.append(dict(dataset="diagnostic_32", configuration=label,
                     mean_action_mse=r["mean_action_mse"], gripper_disagreements=r["gripper_disagreement_steps"],
                     predictions=256, evidence_class="committed_summary_with_per_frame_values"))
    for name, mse in sorted(r["per_sample_action_mse"].items()):
        samples.append(dict(sample=name, configuration=label, teacher_action_mse=mse))
archival = read_json("reports/2026-09-16-weekend-review/data/archival_offline.json")
for r in archival["language_validation"]:
    rows.append(dict(dataset="previous_validation_32", configuration=r["configuration"],
                     mean_action_mse=r["mean_action_mse"], gripper_disagreements=r["gripper_disagreements"],
                     predictions=256, evidence_class="archival_summary"))
write_csv("language_offline.csv", rows)
write_csv("language_diagnostic_per_frame.csv", samples)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
for ax, dataset in zip(axes, ["diagnostic_32", "previous_validation_32"]):
    subset = [r for r in rows if r["dataset"] == dataset]
    x = np.arange(len(subset))
    p = [r["mean_action_mse"] for r in subset]
    ax.bar(x, p, color=["#bc5853", "#329f90", "#577dcc", "#d69335", "#986f99"][:len(x)])
    ax.set_yscale("log"); ax.set_ylim(1e-4, 15)
    ax.set_xticks(x); ax.set_xticklabels([r["configuration"].replace("; ", ";\n") for r in subset], rotation=23, ha="right", fontsize=8)
    for i, r in enumerate(subset):
        ax.text(i, p[i]*1.2, "{:.4g}\ng={}/256".format(p[i], r["gripper_disagreements"]), ha="center", fontsize=8)
    ax.set_ylabel("Mean normalized action MSE vs BF16 (log scale)")
    ax.set_title("32 diagnostic frames" if dataset.startswith("diagnostic") else "32 previously inspected validation frames")
fig.suptitle("Language no-clip reduces offline error; closed-loop recovery is still absent", fontsize=14)
save(fig, "03_language_offline", "g = raw gripper-decision disagreement across 32 x 8 predictions; correlated chunk steps are not independent episodes.\nDiagnostic source includes actual per-frame MSE. Validation values are archival summaries, not newly measured or blind test results.")

vision = archival["vision_diagnostic"]
write_csv("vision_offline.csv", [dict(r, evidence_class="archival_summary", frames=32, predictions=256,
                                    language="W2 G128 no extra clip") for r in vision])
write_csv("vision_combination_validation.csv", [dict(r, evidence_class="archival_summary", frames=32, predictions=256) for r in archival["vision_combination_validation"]])
fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
y = np.arange(len(vision))
for ax, field, title in [(axes[0], "mean_action_mse", "Teacher action MSE (language fixed W2 no-clip)"),
                          (axes[1], "projector_relative_mse", "Projector feature relative MSE")]:
    ax.barh(y, [r[field] for r in vision], color=["#526778", "#259e91", "#bc5853", "#577dcc", "#b77878", "#d69335", "#72a77a"])
    ax.set_yticks(y); ax.set_yticklabels([r["configuration"] for r in vision], fontsize=9)
    ax.invert_yaxis(); ax.set_title(title, fontsize=10)
    ax.set_xlim(0, max(r[field] for r in vision)*1.24)
    for i, r in enumerate(vision):
        ax.text(r[field]+ax.get_xlim()[1]*.018, i+.10, "{:.4g}".format(r[field]), fontsize=8)
fig.suptitle("Vision interventions: DINO is more sensitive; its clipping should not be removed blindly", fontsize=13)
save(fig, "04_vision_offline", "All values: rounded archival summaries, no new GPU measurement. Untargeted vision branch is BF16; DINO/SigLIP are encoders, not cameras.\nG64 uses a recalibrated scale + clip profile. Changes cannot be attributed to group size alone. Local MSE does not establish closed-loop failure.")

historical = read_csv("results/official_quant_validation.csv")
write_csv("historical_sq.csv", historical)
sq = [r for r in historical if r["method"].startswith("SmoothQuant")]
fig, ax = plt.subplots(figsize=(10, 5.2))
p = np.array([float(r["rate"]) for r in sq])
ax.bar(range(len(sq)), p, color=["#648f83", "#648f83", "#648f83", "#bc5853"])
ax.errorbar(range(len(sq)), p, yerr=np.array([p-np.array([float(r["wilson95_low"]) for r in sq]),
              np.array([float(r["wilson95_high"]) for r in sq])-p]), fmt="none", color="black", capsize=3)
for i, r in enumerate(sq):
    ax.text(i, float(r["rate"])+.04, "{}/{}".format(r["successes"], r["episodes"]), ha="center")
ax.set_xticks(range(len(sq))); ax.set_xticklabels([r["method"].replace("SmoothQuant ", "") for r in sq])
ax.set_ylim(0, 1.25); ax.set_ylabel("Historical task success rate")
ax.set_title("SmoothQuant: OLD environment only; current W4A4 baseline is unmeasured", pad=14)
save(fig, "05_sq_historical", "Source: committed historical-log CSV; 20 episodes/configuration; protocol/checkpoint equivalence to current runs is not established.\nNever combine these with current W4/50 results or label 0/20 as the repaired-environment W4A4 baseline. Whiskers use the source Wilson bounds.")

block = read_json("results/awq-block-audit-20260916/block-error-summary.json")
block_rows = []
for condition, values in block["median_relative_mse_action_tokens"].items():
    for layer, vals in zip(block["layers"], values):
        for variant, value in zip(block["variants"], vals):
            block_rows.append(dict(input_condition=condition, layer=layer, variant=variant,
                                   median_action_token_relative_mse=value, selected_frames=3))
write_csv("selected_block_error.csv", block_rows)
fig, axes = plt.subplots(1, len(block["median_relative_mse_action_tokens"]), figsize=(13, 6.2), squeeze=False)
for ax, (condition, values) in zip(axes[0], block["median_relative_mse_action_tokens"].items()):
    arr = np.array(values)
    im = ax.imshow(np.log10(np.maximum(arr, 1e-6)), vmin=-6, vmax=1, cmap="magma", aspect="auto")
    ax.set_yticks(range(len(block["layers"]))); ax.set_yticklabels(block["layers"])
    ax.set_xticks(range(len(block["variants"]))); ax.set_xticklabels([s.replace("w2_", "W2\n").replace("protect_", "keep\n").replace("_", " ") for s in block["variants"]], fontsize=8, rotation=25, ha="right")
    ax.set_title("Input: " + condition); ax.set_ylabel("Selected LLM block")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, "{:.1e}".format(arr[i,j]), ha="center", va="center", fontsize=6.5, color="white")
fig.colorbar(im, ax=list(axes[0]), fraction=.025, pad=.03, label="log10 relative MSE")
fig.suptitle("Selected-case block audit: local reconstruction differs from end-to-end action damage", fontsize=13)
# Manual spacing retains a separate colorbar gutter.
fig.text(.02, .025, "Three deliberately selected frames; five blocks; medians describe local action-token hidden reconstruction.\nThis is not a complete layer ranking. A local rescue does not establish downstream or closed-loop recovery.", fontsize=8)
fig.subplots_adjust(bottom=.28, top=.82, left=.07, right=.88, wspace=.20)
fig.savefig(FIG / "06_selected_block_audit.png"); fig.savefig(FIG / "06_selected_block_audit.svg", metadata={"Date": None})
plt.close(fig)

manifest = {
    "report": "2026-09-16-weekend-review", "kind": "local_evidence_review_no_new_gpu_run",
    "source_hash_format": "Text inputs use UTF-8 bytes with CRLF normalized to LF, matching Git canonical text. Generated files use exact bytes; CSV is written with LF.",
    "source_files": [{"path": p.relative_to(REPO).as_posix(), "sha256": hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
                     for p in sorted(SOURCE_FILES)],
    "generated_files": [{"path": p.relative_to(REPORT).as_posix(), "size_bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(list(FIG.glob("*")) + list(DATA.glob("*.csv")))],
    "not_uploaded": ["source PDF/full extracted text", "model weights", "AWQ profile binaries", "full calibration/trajectory data"],
    "verification_limits": "Hashes bind local committed evidence and generated files. They do not prove semantic correctness of server profiles or identity of original paper results. See evidence classifications in the report."
}
with (REPORT / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
print("Built 6 figures (PNG + SVG), 8 CSV sidecars and manifest; no GPU used.")
