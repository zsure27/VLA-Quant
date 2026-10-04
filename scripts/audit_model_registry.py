"""Reproduce the 2026-10-05 offline registry/data audit. No GPU or external access."""
from __future__ import annotations
import hashlib
import io
import json
import math
import sys
import tarfile
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.model_registry import validate_adapter_shapes
from qvla.state_metadata import tensor_shapes
from qvla.on_policy_capture import observation_hash
from qvla.data_roles import sample_fingerprint
from qvla.paired_metrics import comparison


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def member(tar, suffix):
    matches = [m for m in tar.getmembers() if m.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archived member: {suffix}; got {len(matches)}")
    return tar.extractfile(matches[0]).read()


def main():
    split_path = "results/experiments/p1-data-contract/20260925-107-p1-data-contract/trajectory-split.json"
    split = load(split_path)
    episodes = split["episodes"]
    assert len({r["trajectory_id"] for r in episodes}) == len({r["dataset_order_index"] for r in episodes}) == 432
    assert sum(r["steps"] for r in episodes) == 52970
    by_index = {r["dataset_order_index"]: r for r in episodes}
    profile_path = "results/experiments/p0-foundation-baselines/20260917-107-baseline-validation/awq-baseline-shard-05-14-20260917-121652-1344/awq-profile-metadata.json"
    profile = load(profile_path)[0]
    selected = [r for r in profile["calibration_manifest"]["samples"] if r["file"] in profile["sample_names"]]
    assert len(selected) == profile["num_samples"] == 32
    assert profile["profile_sha256"] == "ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4"
    assert all(profile["sample_sha256"][r["file"]] == r["sha256"] for r in selected)
    assert all(by_index[r["episode"]]["instruction"] == r["instruction"] for r in selected)
    contamination = [{"dataset_order_index": r["episode"], "trajectory_id": by_index[r["episode"]]["trajectory_id"],
                      "later_role": by_index[r["episode"]]["role"], "frame_sha256": r["sha256"]} for r in selected]
    artifacts = {}
    c3_tar = ROOT / "backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/student-state-full.tar.gz"
    student_manifest_path = "results/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/student-state80-manifest.json"
    manifest_raw = (ROOT / student_manifest_path).read_bytes()
    manifest = json.loads(manifest_raw)
    student_hashes = set()
    with tarfile.open(c3_tar) as tar:
        archived_manifest = member(tar, "student-state80/manifest.json")
        assert sha(archived_manifest) == sha(manifest_raw)
        for row in manifest["samples"]:
            raw = member(tar, "student-state80/" + row["file"])
            assert sha(raw) == row["sample_sha256"]
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                fingerprint = observation_hash(data["image"], data["wrist_image"], data["state"], str(data["instruction"].item()))
            assert fingerprint == row["observation_sha256"]
            student_hashes.add(fingerprint)
        raw = member(tar, "exact12l-response-svd-r8-smoothl1-studentstate80-e2e1000/e2e_adapter_state.pt")
        shapes = tensor_shapes(raw); validate_adapter_shapes(shapes, "B0")
        artifacts["B0"] = {"adapter_sha256": sha(raw), "serialized_bytes": len(raw), "tensor_count": len(shapes),
                           "parameters": sum(math.prod(s) for s in shapes.values()), "training_manifest_sha256": sha(manifest_raw)}
    assert len(student_hashes) == len(manifest["samples"]) == 80
    assert {(r["task_id"], r["init_state_index"], r["query_in_episode"]) for r in manifest["samples"]} == {
        (t, reset, q) for t in range(10) for reset in range(4) for q in range(2)}
    lw_tar = ROOT / "backups/experiments/p2-shared-peft/20260930-059-language-w2-all-full.tar"
    command_audit = {}
    with tarfile.open(lw_tar) as tar:
        raw = member(tar, "language-w2-all-r8-e2e1000/e2e_adapter_state.pt")
        shapes = tensor_shapes(raw); validate_adapter_shapes(shapes, "B1")
        artifacts["B1"] = {"adapter_sha256": sha(raw), "serialized_bytes": len(raw), "tensor_count": len(shapes),
                           "parameters": sum(math.prod(s) for s in shapes.values()), "training_manifest_sha256": sha(manifest_raw)}
        training = json.loads(member(tar, "language-w2-all-r8-e2e1000/e2e_training_samples.json"))
        assert {r["sha256"] for r in training["samples"]} == {r["sample_sha256"] for r in manifest["samples"]}
        for phase in ("first50trace", "remaining250"):
            for case, model in (("C0", "A3"), ("C3", "B0"), ("LW", "B1"), ("BF16", "BF16"), ("W4", "A0")):
                command = member(tar, f"eval-{phase}/{case}/command.txt").decode()
                checksums = member(tar, f"eval-{phase}/{case}/CONTRACT_SHA256SUMS.txt").decode()
                if model in artifacts:
                    assert artifacts[model]["adapter_sha256"] in checksums
                    assert ("candidate-adapter.pt" if model == "B0" else "language-w2-all-r8-e2e1000/e2e_adapter_state.pt") in command
                command_audit[f"{phase}/{model}"] = {"command_sha256": sha(command.encode()), "adapter_binding": "passed" if model in artifacts else "not applicable"}
    dev = load("results/experiments/p2-shared-peft/20260927-107-p2-router-dev/router-dev-sample-manifest.json")
    dev_dir = ROOT / "backups/experiments/p2-shared-peft/20260927-107-router-dev/router-dev-frames-p10-p30-p50-p70-p90"
    dev_hashes = set()
    with tarfile.open(dev_dir.parent / "full.tar") as tar:
        for row in dev["samples"]:
            original = by_index[row["dataset_order_index"]]
            assert original["role"] == row["role"] == "router_dev" and original["trajectory_id"] == row["trajectory_id"]
            raw = member(tar, "router-dev-frames-p10-p30-p50-p70-p90/" + row["file"])
            assert sha(raw) == row["sha256"]
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                dev_hashes.add(observation_hash(data["image"], data["wrist_image"], data["state"], str(data["instruction"].item())))
    assert not student_hashes & dev_hashes
    source = "results/experiments/p2-shared-peft/20260930-059-language-w2-all/paired-spatial300-analysis.json"
    paired = load(source)
    aliases = {"C0": "A3", "C3": "B0", "LW": "B1", "BF16": "BF16", "W4": "A0"}
    rows = [{**{k: v for k, v in r.items() if k not in aliases}, **{new: r[old] for old, new in aliases.items()}} for r in paired["rows"]]
    assert {(r["task_id"], r["init_state_index"]) for r in rows} == {(t, i) for t in range(10) for i in range(20, 50)}
    assert not {(r["task_id"], r["init_state_index"]) for r in rows} & {(r["task_id"], r["init_state_index"]) for r in manifest["samples"]}
    pairs = (("B0", "A3"), ("B1", "A3"), ("B1", "B0"), ("B0", "BF16"), ("B0", "A0"), ("B1", "BF16"), ("B1", "A0"))
    output = {"classification": "offline archival audit; development cohort, not independent holdout",
              "split_counts": dict(Counter(r["role"] for r in episodes)),
              "awq32_later_split_overlap": dict(Counter(r["later_role"] for r in contamination)),
              "awq32_selected_trajectories": contamination, "whole_pipeline_holdout_status": "NOT_CLEAN: 6/75 used by frozen W4 calibration",
              "student_state": {"observations": 80, "source_episodes": 40, "queries_per_episode": 2,
                                "raw_sha_and_observation_sha_passed": True, "router_dev_observation_overlap": 0,
                                "paired300_training_reset_overlap": 0},
              "adapters": artifacts, "archived_command_bindings": command_audit,
              "paired300_successes": {m: sum(r[m] for r in rows) for m in aliases.values()},
              "comparisons": {f"{a}_vs_{b}": comparison(rows, a, b) for a, b in pairs},
              "source_sha256": {p: sha((ROOT/p).read_bytes()) for p in (split_path, profile_path, student_manifest_path, source)},
              "limitations": ["No new GPU reload/zero-residual test", "Other frozen profiles require their own calibration intersection audit",
                              "Inherited Response-SVD calibration80 source trajectory manifest requires next-boot verification; file hashes alone do not prove train-only roles",
                              "Metadata shape inspection does not certify tensor finite values", "Base checkpoint training exclusion not established",
                              "Old 411/412 and 430/431 remain distinct source cohorts", "H17 from B1 is after some adapter layers: not a router feature before expert selection"]}
    target = ROOT / "results/experiments/p2-shared-peft/20261005-model-registry-audit/audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("awq32_later_split_overlap", "adapters", "paired300_successes", "comparisons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
