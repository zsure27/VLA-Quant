"""Verify transferred results and fixed B3 inputs before a new GPU session.

Read-only on all historical materials. Only the new receipt directory is written.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import time


def verify_manifest(root, filename, expected_count, cache):
    rows = (root / filename).read_text().splitlines()
    if len(rows) != expected_count:
        raise ValueError(f"{filename}: expected {expected_count} entries, got {len(rows)}")
    for row in rows:
        digest, name = row.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Archive path missing/outside archive: {name}")
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for part in iter(lambda: stream.read(4 * 1024**2), b""):
                    value.update(part)
            cache[key] = value.hexdigest()
        if cache[key] != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    return {"entries": len(rows), "all_sha256_pass": True,
            "manifest_sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "materials", "archive", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Never overwrite a prior receipt")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = {"version": "046-quant-finetuning-resume-v1-20261015",
              "time_utc": datetime.now(timezone.utc).isoformat(),
              "hostname": socket.gethostname(), "gpu_experiment_started": False}
    try:
        if socket.gethostname() != a.expected_hostname:
            raise ValueError("Wrong current instance")
        if shutil.disk_usage(a.archive).free < 12 * 1024**3:
            raise ValueError("Need 12GiB persistent free before first GPU experiment")
        cache = {}
        result["archive"] = {name: verify_manifest(a.archive, name, count, cache)
                             for name, count in (("SHA256SUMS.txt", 22244),
                                                 ("RESULTS_SHA256SUMS.txt", 15389))}
        sys.path.insert(0, str(a.source_root))
        sys.path.insert(0, str(a.source_root / "diagnostics"))
        from scripts.train_extended_peft import materials_contract
        from qvla.baseline_contract import checkpoint_identity
        from qvla.run_eval_official_quant import load_profiles
        from qvla.extended_peft import sha
        m = json.loads(a.materials.read_text())
        paths, calibration, audit = materials_contract(m, "B3")
        _, meta = load_profiles([Path(m["w2_g128"])], "awq")
        identity = checkpoint_identity(m["checkpoint"])
        if identity != meta["checkpoint_identity"]:
            raise ValueError("Full checkpoint differs from frozen AWQ profile")
        result.update({"gate": "PASS_TRANSFERRED_ARCHIVE_AND_B3_INPUTS",
                       "training_observations": len(paths), "calibration_trajectories": len(calibration),
                       "input_audit": audit, "checkpoint_identity": identity,
                       "materials_sha256": sha(a.materials),
                       "archive_root": str(a.archive), "source_root": str(a.source_root),
                       "free_bytes": shutil.disk_usage(a.archive).free})
    except Exception as exc:
        result.update({"gate": "FAIL_RESUME_CONTENT_GATE", "error_type": type(exc).__name__, "error": str(exc)})
    result["elapsed_seconds"] = time.monotonic() - start
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("gate", "hostname", "elapsed_seconds")}))
    if result["gate"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
