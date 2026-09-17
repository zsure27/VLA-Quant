"""Read-only CPU audit of the frozen AWQ profiles; no model load or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def audit(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    if d.get("method") != "awq":
        raise ValueError("Expected an AWQ profile")
    entries = d["entries"]
    rows = []
    for name, entry in entries.items():
        shape = entry["shape"]
        count = 1
        for dim in shape:
            count *= dim
        branch = "language" if name.startswith("language_model.") else (
            "DINO" if name.startswith("vision_backbone.featurizer.") else "SigLIP")
        rows.append(dict(module=name, branch=branch, shape=shape, weight_parameters=count,
                         has_clip=entry.get("clip_max") is not None,
                         has_input_scale=entry.get("input_scale") is not None,
                         calibration_rows=entry.get("calibration_rows")))
    metadata_fields = ("method", "bits", "activation_bits", "group_size", "num_samples",
                       "seed", "algorithm", "llm_attention", "format_version", "contract_id",
                       "checkpoint_identity", "sample_sha256", "sample_names", "calibration_manifest",
                       "implementation_sources", "official_sources", "model_source_sha256")
    result = {key: d[key] for key in metadata_fields if key in d}
    result.update(profile=str(path), entry_count=len(entries), module_rows=rows,
                  block_scale_count=len(d.get("block_scales", {})),
                  note="Profile search metadata; CPU read only. Does not establish closed-loop accuracy or packed compression.")
    result["scope_summary"] = {}
    for branch in ("language", "DINO", "SigLIP"):
        values = [row for row in rows if row["branch"] == branch]
        result["scope_summary"][branch] = dict(targets=len(values),
            weight_parameters=sum(row["weight_parameters"] for row in values),
            clip_targets=sum(row["has_clip"] for row in values))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result["profile_sha256"] = digest.hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = [audit(path) for path in args.profile]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{key: row.get(key) for key in ("profile", "bits", "group_size", "num_samples", "entry_count", "scope_summary")} for row in result]))
