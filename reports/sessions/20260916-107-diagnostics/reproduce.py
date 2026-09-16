"""Plot actual stage-rescue/control evidence copied from instance 107.

Reads results/107-diagnostics-20260916; never connects to a server.
Partial batches must retain an explicit partial status in README, not a completion claim.
"""
import csv
import hashlib
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = REPO / "results/107-diagnostics-20260916"
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "svg.hashsalt": "vla-107-20260916",
                     "axes.spines.top": False, "axes.spines.right": False})


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def output_csv(name, rows):
    with (DATA / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def save(fig, name, note):
    fig.text(.02, .02, note, fontsize=8)
    fig.tight_layout(rect=[0, .13, 1, .94])
    fig.savefig(FIG / (name + ".png"), dpi=180)
    fig.savefig(FIG / (name + ".svg"), metadata={"Date": None})
    svg = FIG / (name + ".svg")
    svg.write_bytes(svg.read_bytes().replace(b"\r\n", b"\n"))
    plt.close(fig)


configurations = {}
if (RAW / "baseline/metrics.json").exists():
    configurations["W2 no-clip baseline"] = read(RAW / "baseline/metrics.json")
for path in sorted(RAW.glob("awq-language-stage-rescue-*/stage-*/metrics.json")):
    scope = read(path.parent / "scope.json")
    name = "W4 blocks " + ",".join(map(str, scope["w4_layers"]))
    configurations[name] = read(path)
summary_rows, per_sample, per_dim = [], [], []
baseline = {r["sample"]: r for r in configurations.get("W2 no-clip baseline", [])}
for name, records in configurations.items():
    if len(records) != 32 or {r["sample"] for r in records} != set(baseline):
        raise RuntimeError("Missing or unpaired 32-frame evidence: " + name)
    errors = np.array([r["normalized_action"]["mse"] for r in records])
    differences = np.array([r["normalized_action"]["mse"]-baseline[r["sample"]]["normalized_action"]["mse"] for r in records])
    boot = np.random.default_rng(7).choice(differences, size=(10000, 32), replace=True).mean(1)
    low, high = np.quantile(boot, [.025, .975])
    summary_rows.append(dict(configuration=name, frames=32, chunk_steps=8,
        mean_teacher_action_mse=float(errors.mean()), median_teacher_action_mse=float(np.median(errors)),
        max_teacher_action_mse=float(errors.max()), gripper_disagreements=round(sum(r["raw_gripper_disagreement"]*8 for r in records)),
        improved_frames=int((differences<0).sum()), worsened_frames=int((differences>0).sum()),
        mean_paired_mse_difference=float(differences.mean()), exploratory_bootstrap_low=float(low),
        exploratory_bootstrap_high=float(high), evidence_class="fetched_actual_metrics_development"))
    for r in records:
        per_sample.append(dict(configuration=name, sample=r["sample"],
            teacher_action_mse=r["normalized_action"]["mse"],
            baseline_teacher_action_mse=baseline[r["sample"]]["normalized_action"]["mse"],
            gripper_disagreements=round(r["raw_gripper_disagreement"]*8)))
        for i, value in enumerate(r.get("normalized_rmse_per_dim", [])):
            per_dim.append(dict(configuration=name, sample=r["sample"], dimension=i, normalized_rmse=value))
if summary_rows:
    output_csv("stage_summary.csv", summary_rows); output_csv("stage_per_frame.csv", per_sample)
    if per_dim: output_csv("stage_per_dimension.csv", per_dim)
    if per_dim:
        names = list(configurations)
        dimension_mse = [[np.mean([r["normalized_rmse"]**2 for r in per_dim if r["configuration"] == name and r["dimension"] == i]) for i in range(7)] for name in names]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        im = ax.imshow(np.sqrt(dimension_mse), aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(names)))
        short = []
        for name in names:
            if name.startswith("W4 blocks "):
                layers = name.replace("W4 blocks ", "").split(",")
                short.append("W4 blocks {}-{}".format(layers[0],layers[-1]))
            else: short.append(name)
        ax.set_yticklabels(short)
        ax.set_xticks(range(7)); ax.set_xticklabels(["dim {}".format(i) for i in range(7)])
        for i, values in enumerate(dimension_mse):
            for j,v in enumerate(values): ax.text(j,i,"{:.3f}".format(np.sqrt(v)),ha="center",va="center",fontsize=8)
        fig.colorbar(im, ax=ax, label="Normalized teacher-action RMSE")
        ax.set_title("Paired stage rescue: error differs across the seven action dimensions")
        save(fig, "04_action_dimensions", "Aggregate RMSE = sqrt(mean of squared per-frame RMSE), each frame predicts eight steps.\nDimension 6 is the continuous gripper output; this heatmap is not physical displacement or a discrete success metric.")
    labels = [r["configuration"].replace("W4 blocks ", "W4\n").replace(",", " ") for r in summary_rows]
    labels = ["W4 {}-{}".format(label.split()[1],label.split()[-1]) if label.startswith("W4") else label for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(range(len(labels)), [r["mean_teacher_action_mse"] for r in summary_rows], color="#389d8a")
    axes[1].bar(range(len(labels)), [r["gripper_disagreements"] for r in summary_rows], color="#cd8c3f")
    for ax in axes:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean normalized action MSE vs BF16")
    axes[1].set_ylabel("Gripper disagreement / 256 predictions")
    fig.suptitle("107: language-stage W4 rescue on W2 no-clip backbone")
    save(fig, "01_stage_rescue", "32 previously inspected development trajectories, one frame each; 8-step chunks. Vision BF16.\nW4 blocks retain their own scale/clip; other language blocks use W2 without extra clip. No PEFT or closed-loop claim.")
    candidates = list(configurations)[1:]
    if candidates:
        fig, axes = plt.subplots(1, len(candidates), figsize=(max(6, 3.6*len(candidates)), 4.8), squeeze=False)
        for ax, name in zip(axes[0], candidates):
            rows = [r for r in per_sample if r["configuration"] == name]
            x, y = [r["baseline_teacher_action_mse"] for r in rows], [r["teacher_action_mse"] for r in rows]
            ax.scatter(x, y, s=22, color="#577dcc")
            lim = max(x+y)*1.05; ax.plot([0,lim],[0,lim], "--", color="gray")
            ax.set_xlim(0,lim); ax.set_ylim(0,lim)
            layerlist = name.replace("W4 blocks ", "").split(",")
            ax.set_title("W4 blocks {}-{}".format(layerlist[0], layerlist[-1]))
            ax.set_xlabel("W2 baseline MSE"); ax.set_ylabel("Rescue MSE")
        save(fig, "02_paired_stage_frames", "Each point is one actual paired development frame; points below the diagonal improved.\nExploratory statistics do not adjust prior method selection or establish closed-loop restoration.")

sq_records, sq_summary, feature_rows = [], [], []
for path in sorted(RAW.glob("sq-controls-recheck-*/*/metrics.json")):
    values = read(path)
    if len(values) != 8: raise RuntimeError("Eight control frames required")
    for r in values:
        sq_records.append(dict(configuration=path.parent.name, sample=r["sample"],
            action_mse=r["normalized_action"]["mse"], gripper_disagreements=round(r["raw_gripper_disagreement"]*8)))
        for key, metric in r["features"].items():
            match = re.fullmatch(r"language_model.model.layers.(\d+)@0/(.+)", key)
            if match:
                feature_rows.append(dict(configuration=path.parent.name, sample=r["sample"],
                    block=int(match[1]), token_partition=match[2],
                    relative_mse=metric["relative_mse"], cosine=metric["cosine"],
                    max_abs_error=metric["max_abs_error"]))
    sq_summary.append(dict(configuration=path.parent.name, frames=8,
        mean_action_mse=float(np.mean([r["normalized_action"]["mse"] for r in values])),
        max_action_mse=max(r["normalized_action"]["mse"] for r in values),
        gripper_disagreements=round(sum(r["raw_gripper_disagreement"]*8 for r in values))))
if sq_records:
    output_csv("sq_control_per_frame.csv", sq_records); output_csv("sq_control_summary.csv", sq_summary)
    fig, ax = plt.subplots(figsize=(9,4.8))
    for name in sorted({r["configuration"] for r in sq_records}):
        rows = [r for r in sq_records if r["configuration"] == name]
        ax.plot(range(8), [r["action_mse"] for r in rows], marker="o", label=name)
    ax.axhline(1e-4, color="gray", linestyle="--", label="Original engineering screen 1e-4")
    ax.set_yscale("symlog", linthresh=1e-7); ax.set_xlabel("Paired development frame index (64-71)")
    ax.set_ylim(0, max(r["action_mse"] for r in sq_records)*1.6)
    ax.set_ylabel("Normalized action MSE vs cached BF16"); ax.legend(fontsize=8)
    ax.set_title("107: SQ BF16 controls; measurement completion is separate from gate passing")
    save(fig, "03_sq_control", "Eight original development frames; no W/A low-bit quantization, no rollout, no training.\nRepeat is the numerical control. A failed smooth-only gate must be diagnosed before the W4A4 baseline is accepted.")

if feature_rows:
    output_csv("sq_features_per_frame.csv", feature_rows)
    parts = ["image_main", "image_wrist", "proprio", "text", "action_readout"]
    names = sorted({r["configuration"] for r in feature_rows if r["configuration"] != "repeat"})
    fig, axes = plt.subplots(1, len(names), figsize=(max(7, 5*len(names)), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        values = [[np.mean([r["relative_mse"] for r in feature_rows if r["configuration"] == name
                    and r["block"] == block and r["token_partition"] == part])
                   for block in range(32)] for part in parts]
        im = ax.imshow(np.log10(np.maximum(values, 1e-12)), aspect="auto", cmap="magma", vmin=-12, vmax=-1)
        ax.set_yticks(range(len(parts))); ax.set_yticklabels(parts)
        ax.set_xlabel("Language block"); ax.set_title(name)
        fig.colorbar(im, ax=ax, label="log10(mean relative MSE)")
    save(fig, "06_sq_token_partition", "Eight development inputs, BF16 weight/activation controls. Floor 1e-12 is for display only.\nA layer/partition association is not causal proof; detailed per-frame values are in CSV, including unclamped floating-point cosines.")

family_rows, family_summary = [], []
for path in sorted(RAW.glob("awq-family-precision-*/*/metrics.json")):
    records = read(path)
    if len(records) != 32 or {r["sample"] for r in records} != set(baseline):
        raise RuntimeError("Family rescue samples are missing or unpaired")
    scope = read(path.parent / "scope.json")
    for r in records:
        family_rows.append(dict(configuration=path.parent.name, sample=r["sample"],
            teacher_action_mse=r["normalized_action"]["mse"],
            baseline_teacher_action_mse=baseline[r["sample"]]["normalized_action"]["mse"],
            gripper_disagreements=round(r["raw_gripper_disagreement"]*8)))
    family_summary.append(dict(configuration=path.parent.name, frames=32,
        mean_teacher_action_mse=float(np.mean([r["normalized_action"]["mse"] for r in records])),
        gripper_disagreements=round(sum(r["raw_gripper_disagreement"]*8 for r in records)),
        rescue_coordinates=scope["rescue_coordinates"],
        evidence_class="fetched_actual_development_family_precision_no_peft"))
if family_summary:
    output_csv("family_summary.csv", family_summary); output_csv("family_per_frame.csv", family_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    labels = ["W2 baseline"] + [r["configuration"]+" W4" for r in family_summary]
    axes[0].bar(labels, [summary_rows[0]["mean_teacher_action_mse"]] + [r["mean_teacher_action_mse"] for r in family_summary], color="#577dcc")
    axes[1].bar(labels, [summary_rows[0]["gripper_disagreements"]] + [r["gripper_disagreements"] for r in family_summary], color="#cd8c3f")
    axes[0].set_ylabel("Mean normalized action MSE vs BF16")
    axes[1].set_ylabel("Gripper disagreements / 256")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Blocks 8-15: attention vs MLP bit rescue at fixed W2 coordinates")
    save(fig, "07_family_precision", "32 previously inspected development frames; vision BF16, every language block retains W2 scales, no extra clip.\nSelected linears use 4 bits: attention 32 vs MLP 24; parameter/storage budgets differ. Not a searched W4 baseline or PEFT result.")


def read_rollout(directory):
    files = list(directory.glob("EVAL-*.txt"))
    if len(files) != 1 or (directory / "exit-code.txt").read_text().strip() != "0":
        raise RuntimeError("Missing or failed closed-loop evidence: " + str(directory))
    text = files[0].read_text(encoding="utf-8")
    manifests = [json.loads(x) for x in re.findall(r"^EPISODE_MANIFEST (.+)$", text, re.M)]
    successes = [x == "True" for x in re.findall(r"^Success: (True|False)$", text, re.M)]
    if len(manifests) != 10 or len(successes) != 10 or "Episode error:" in text:
        raise RuntimeError("Expected ten completed paired initial states")
    return manifests, successes, files[0]


closed_summary = []
closed_batches = list(RAW.glob("language-stage-rescue10-*"))
if closed_batches:
    if len(closed_batches) != 1 or not (closed_batches[0] / "complete.json").exists():
        raise RuntimeError("Closed-loop batch has not completed")
    old = REPO / "results/awq-visual-diagnostics-20260915/raw/diagnostic-20260915-215923"
    runs = [("BF16 (historical paired control)", old / "bf16"),
            ("W4 language + vision (historical)", old / "w4"),
            ("W2 language no-clip (historical)", old / "language")]
    runs += [("Language W4 blocks " + p.name.replace("stage-", ""), p)
             for p in sorted(closed_batches[0].glob("stage-*"))]
    paired_reference, closed_rows, outcomes = None, [], []
    for name, path in runs:
        manifests, successes, source = read_rollout(path)
        if paired_reference is None: paired_reference = manifests
        if manifests != paired_reference:
            raise RuntimeError("State hashes or seeds differ: " + name)
        outcomes.append(successes)
        closed_summary.append(dict(configuration=name, successes=sum(successes), episodes=10,
            success_rate=sum(successes)/10, paired_manifest_equal=True,
            source=source.relative_to(REPO).as_posix(),
            source_sha256=hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            evidence_class="actual_paired_development_screen_no_peft_no_packing"))
        for m, s in zip(manifests, successes):
            closed_rows.append(dict(configuration=name, **m, success=s))
    output_csv("closed_loop_summary.csv", closed_summary)
    output_csv("closed_loop_per_task.csv", closed_rows)
    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.imshow(outcomes, aspect="auto", cmap=ListedColormap(["#ce6767", "#459f80"]), vmin=0, vmax=1)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r["configuration"] + ": {}/10".format(r["successes"]) for r in closed_summary], fontsize=8)
    ax.set_xticks(range(10)); ax.set_xlabel("LIBERO Spatial task, initial state index 0")
    for i, row in enumerate(outcomes):
        for j, v in enumerate(row): ax.text(j, i, "P" if v else "F", ha="center", va="center", color="white")
    ax.set_title("107: mid-language W4 rescue in a W2 backbone, matched state hashes and seeds")
    save(fig, "05_closed_loop_rescue", "Ten previously inspected development initial states; the historical controls are not additional independent trials.\nNew candidates keep vision BF16. Mixed precision sensitivity intervention, zero trained parameters; no uniform-W2 restoration claim.")

manifest = {"session": "20260916-107-diagnostics", "kind": "gpu_offline_and_closed_loop_diagnostics",
    "source_hash_format": "Committed text sources are hashed with CRLF normalized to LF; generated CSV/SVG use LF, binary images use exact bytes.",
    "analysis_source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    "numpy": np.__version__, "matplotlib": matplotlib.__version__,
    "raw_files": [{"path": p.relative_to(REPO).as_posix(), "sha256": hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
                  for p in sorted(RAW.rglob("*")) if p.is_file() and p.suffix != ".npz"],
    "generated_files": [{"path": p.relative_to(HERE).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(list(DATA.glob("*.csv"))+list(FIG.glob("*"))+list((HERE / "video-analysis").glob("*")))],
    "backup_metadata": [{"path": p.relative_to(HERE).as_posix(), "sha256": hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
                        for p in sorted((HERE / "backup").glob("*")) if p.is_file() and p.suffix != ".md"],
    "scope": "Actual offline teacher agreement and paired development closed-loop screening. No synthetic data, trained PEFT or packed kernel result.",
    "summary": summary_rows, "sq_summary": sq_summary, "closed_loop_summary": closed_summary,
    "family_summary": family_summary}
with (HERE / "manifest.json").open("w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(manifest, indent=2)+"\n")
print(json.dumps({"stage_summary": summary_rows, "sq_summary": sq_summary}, indent=2))
