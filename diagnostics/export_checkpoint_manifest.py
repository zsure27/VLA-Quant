"""Export hashes and restore paths without copying checkpoint/profile tensors into Git."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/qvla-repro"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = [args.root / "artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt",
                args.root / "artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt",
                args.root / "artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt"]
    profile = torch.load(profiles[0], map_location="cpu", weights_only=True)
    checkpoint = args.root / "models/openvla-7b-oft-finetuned-libero-spatial"
    manifest = {
        "checkpoint_restore_path": str(checkpoint),
        "checkpoint_identity": profile["checkpoint_identity"],
        "checkpoint_identity_provenance": "Recorded by calibration and recomputed/equality-checked by every official evaluation in this session",
        "checkpoint_file_sizes": {name: (checkpoint / name).stat().st_size for name in profile["checkpoint_identity"]["files"]},
        "profiles": [{"restore_path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in profiles],
        "calibration_sample_hashes": profile.get("sample_sha256"),
        "cloud_backup_scope": "Git contains manifests, code, logs and plots; model/adapter/profile tensors remain on the persistent disk and are NOT claimed as a second cloud copy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("CHECKPOINT_AND_PROFILE_MANIFEST_EXPORTED")


if __name__ == "__main__":
    main()
