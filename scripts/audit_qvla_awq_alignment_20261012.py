"""Recompute archived A0 metrics and check source provenance, without torch/GPU.

This does not certify QVLA paper reproduction, numerical kernel parity, or a
new evaluation. Only named historical development files are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = Path("reports/experiments/p0-foundation-baselines/20260917-107-baseline-validation/awq-profile-metadata.json")
PAIRS = Path("reports/experiments/p0-foundation-baselines/20260917-107-baseline-continuation/data/paired_episodes.csv")
SOURCES = ("official_quant_adapter.py", "awq_block.py", "action_jacobian_batch.py", "runtime_contract.py")
OFFICIAL = {
    "awq_auto_scale": "awq/quantize/auto_scale.py",
    "awq_auto_clip": "awq/quantize/auto_clip.py",
    "awq_quantizer": "awq/quantize/quantizer.py",
}


def digest(path: Path, normalize_newlines=False):
    raw = path.read_bytes()
    if normalize_newlines:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def verify_source(path: Path, expected: str):
    raw, normalized = digest(path), digest(path, True)
    if expected not in (raw, normalized):
        raise ValueError(f"Source differs from frozen metadata: {path}")
    return {"path": str(path), "expected": expected, "raw_sha256": raw,
            "lf_sha256": normalized, "match": "exact_bytes" if raw == expected else "lf_normalized"}


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Success label must be explicitly True or False")
    return value == "True"


def load_pairs(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(r["task_id"]), int(r["init_state_index"])) for r in rows]
    expected = {(t, s) for t in range(10) for s in range(50)}
    if len(keys) != 500 or len(set(keys)) != 500 or set(keys) != expected:
        raise ValueError("Require all 500 unique task/reset pairs, with no repetitions")
    for row in rows:
        boolean(row["bf16_success"])
        boolean(row["w4_success"])
        if row["bf16_episode_errors"] != "[]" or row["w4_episode_errors"] != "[]":
            raise ValueError("Archived baseline contains program errors")
        if len(row["init_state_sha256"]) != 64:
            raise ValueError("Missing initial-state hash")
    return rows


def metrics(rows):
    bf = [boolean(r["bf16_success"]) for r in rows]
    w4 = [boolean(r["w4_success"]) for r in rows]
    rescue = sum(not b and a for b, a in zip(bf, w4))
    broken = sum(b and not a for b, a in zip(bf, w4))
    return {"episodes": len(rows), "BF16_success": sum(bf), "A0_success": sum(w4),
            "BF16_percentage": 100 * sum(bf) / len(rows),
            "A0_percentage": 100 * sum(w4) / len(rows),
            "A0_vs_BF16_rescue": rescue, "A0_vs_BF16_break": broken,
            "net": rescue - broken}


def audit(repo: Path, official_root: Path | None):
    records = json.loads((repo / META).read_text(encoding="utf-8"))
    matches = [r for r in records if r.get("method") == "awq" and r.get("bits") == 4]
    if len(matches) != 1:
        raise ValueError("Expected exactly one archived W4 profile metadata record")
    meta = matches[0]
    required = {"activation_bits": 16, "group_size": 128, "num_samples": 32,
                "algorithm": "awq_official_llama_vision_adapter_v1", "entry_count": 422}
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError(f"Archived recipe changed: {key}")
    expected_scope = {"language": 224, "DINO": 93, "SigLIP": 105}
    if {k: v["targets"] for k, v in meta["scope_summary"].items()} != expected_scope:
        raise ValueError("Declared quantization scope differs")
    names = (repo / "configs/qvla-connected-422.txt").read_text().splitlines()
    if len(names) != 422 or len(set(names)) != 422:
        raise ValueError("Target list has wrong count or duplicates")
    if (sum(n.startswith("language_model.") for n in names),
        sum(n.startswith("vision_backbone.featurizer.") for n in names),
        sum(n.startswith("vision_backbone.fused_featurizer.") for n in names)) != (224, 93, 105):
        raise ValueError("Target list branches differ")
    if sum(n.endswith(".patch_embed.proj") for n in names) != 2:
        raise ValueError("Patch-convolution scope differs")
    implementation = {}
    for name in SOURCES:
        implementation[name] = verify_source(repo / "qvla" / name, meta["implementation_sources"][name])
    official = {}
    if official_root is not None:
        for key, rel in OFFICIAL.items():
            official[key] = verify_source(official_root / rel, meta["official_sources"][key]["sha256"])
    rows = load_pairs(repo / PAIRS)
    full = metrics(rows)
    subset = metrics([r for r in rows if int(r["init_state_index"]) >= 5])
    if (full["BF16_success"], full["A0_success"], full["A0_vs_BF16_rescue"], full["A0_vs_BF16_break"]) != (487, 486, 9, 10):
        raise ValueError("Canonical full baseline recomputation differs")
    if (subset["episodes"], subset["BF16_success"], subset["A0_success"],
        subset["A0_vs_BF16_rescue"], subset["A0_vs_BF16_break"]) != (450, 440, 439, 8, 9):
        raise ValueError("Reset5-49 recomputation differs")
    return {
        "gate": "PASS_ARCHIVED_A0_PROVENANCE_AND_METRICS",
        "classification": "historical same-task development, not a new GPU run",
        "inputs": {str(p): digest(repo / p) for p in (META, PAIRS, Path("configs/qvla-connected-422.txt"))},
        "profile_sha256_from_archived_metadata": meta["profile_sha256"],
        "profile_binary_rehashed_this_run": False,
        "implementation_source_checks": implementation,
        "official_primitive_source_checks": official,
        "scope": expected_scope, "full_reset0_49": full, "filtered_reset5_49": subset,
        "paper_table2": {"source": "https://arxiv.org/html/2602.03782v1#S4.T2",
                         "OpenVLA_OFT_Spatial": {"FP": 97.6, "AWQ_W4A16": 93.0},
                         "four_suite_average": {"FP": 97.1, "AWQ_W4A16": 92.5}},
        "unresolved": [
            "QVLA table2 AWQ recipe/calibration/actual command and checkpoint revision are not publicly fixed.",
            "Our vision adaptation and per-episode paired seeding are not proven to match the paper.",
            "This audit checks source bytes and archived declared coverage, not tensor-level quantization or GPU parity.",
            "Packed W4/W2 storage, kernels and latency are not certified.",
            "Reset5-49 does not automatically exclude offline PTQ calibration exposure.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--official-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Never overwrite an audit receipt")
    result = audit(a.repo.resolve(), a.official_root.resolve() if a.official_root else None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gate", "full_reset0_49", "filtered_reset5_49")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

