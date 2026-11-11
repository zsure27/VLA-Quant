"""Explicit native-JAX checkpoint identity plugin; no numerical-gate waiver."""
from pathlib import Path
import json
from qvla import baseline_contract
from qvla.infra.common import digest, canonical

_original = baseline_contract.checkpoint_identity


def checkpoint_identity(path):
    root = Path(path).resolve()
    receipt = root / "DOWNLOAD_VERIFICATION.json"
    if not (root / "params").is_dir() or not receipt.is_file():
        return _original(path)
    record = json.loads(receipt.read_text())
    if (record.get("source") != "gs://openpi-assets/checkpoints/pi05_libero/"
            or record.get("objects") != 16 or len(record.get("files", [])) != 16):
        raise ValueError("Unexpected native checkpoint manifest")
    rows = {}
    total = 0
    for row in record["files"]:
        rel = Path(row["path"])
        item = root / rel
        if (rel.is_absolute() or ".." in rel.parts or item.is_symlink()
                or not item.resolve().is_relative_to(root)
                or item.stat().st_size != row["bytes"] or digest(item) != row["sha256"]):
            raise ValueError("Native checkpoint content/path mismatch")
        rows[rel.as_posix()] = row["sha256"]
        total += row["bytes"]
    if total != record["bytes"]:
        raise ValueError("Native checkpoint total mismatch")
    return {"files": rows, "sha256": canonical(rows), "schema": "openpi-native-manifest-v1"}


baseline_contract.checkpoint_identity = checkpoint_identity
