"""Archive completed B3 Z/S training only; active rollouts are excluded."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "session", "archive", "analysis"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname or a.archive.exists():
        raise ValueError("Wrong instance or archive already exists; preserve prior attempt")
    pair = json.loads((a.session / "train-1000-pair-gate.json").read_text())
    if pair.get("gate") != "PASS_B3_INITIALIZATION_TRAIN_PAIR":
        raise ValueError("Training pair gate missing")
    if shutil.disk_usage(a.session).free < 6 * 1024**3:
        raise ValueError("Preserve at least 6GiB reserve")
    m = json.loads((a.control / "materials-b3-initialization-v1.json").read_text())
    a.archive.mkdir(parents=True)
    origins = {}
    def add_file(source, relative):
        source = Path(source)
        destination = a.archive / relative
        if not source.is_file() or destination.exists():
            raise ValueError(f"Missing source or duplicate archive entry: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        origins[relative] = {"source": str(source), "sha256": digest(destination), "bytes": destination.stat().st_size}
    def add_tree(source, relative):
        source = Path(source)
        if not source.is_dir():
            raise ValueError(f"Missing completed directory: {source}")
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add_file(path, relative + "/" + path.relative_to(source).as_posix())
    for variant in ("standard-zero", "response-svd"):
        for steps in (10, 1000):
            path = a.session / variant / f"steps-{steps}"
            artifact = json.loads((path / "artifact.json").read_text())
            if artifact.get("gate") != ("PASS_SMOKE10" if steps == 10 else "PASS_TRAIN_CONTRACT"):
                raise ValueError("Cannot archive incomplete training as complete")
            add_tree(path, f"payload/training/{variant}/steps-{steps}")
    completed = [f"train-{n}-{v}" for n in (10, 1000) for v in ("standard-zero", "response-svd")]
    completed += ["audit-training-10", "audit-training-1000"]
    for name in completed:
        path = a.session / "stage-records" / name
        if (path / "exit-code.txt").read_text().strip() != "0":
            raise ValueError("Completed wrapper must have exit0")
        add_tree(path, "payload/wrapper-records/" + name)
    for name in ("train-10-pair-gate.json", "train-1000-pair-gate.json", "resume-content-gate.json",
                 "actual-observation-repeatability.json", "actual-observation-repeatability-v2.json"):
        add_file(a.session / name, "payload/gates/" + name)
    source_manifest = json.loads((a.control / "source-v2-SHA256.json").read_text())
    for name, expected in source_manifest.items():
        source_file = a.control / "source-v2" / name
        if digest(source_file) != expected:
            raise ValueError("Frozen source mismatch: " + name)
        add_file(source_file, "payload/source-v2/" + name)
    for path in sorted(a.control.iterdir()):
        if (path.is_file() and path.suffix in (".json", ".py", ".gz", ".log")
                and path.name not in ("B3-supervisor-v1.log", "archive-training-v1.log")):
            add_file(path, "payload/control/" + path.name)
    for name in ("first50-plan.json", "runtime-lock.json"):
        add_file(a.control / "B3-plans-v1" / name, "payload/control/B3-plans-v1/" + name)
    add_tree(m["student80"], "payload/inputs/student80")
    add_tree(m["calibration80"], "payload/inputs/calibration80")
    add_file(m["trajectory_split"], "payload/inputs/trajectory-split.json")
    for name in ("artifact.json", "initial-adapter.pt", "teacher-targets.pt", "teacher-input-hashes.json"):
        add_file(Path(m["shared_b3_smoke_dir"]) / name, "payload/inputs/inherited-pretraining-cache/" + name)
    add_tree(a.analysis, "payload/analysis")
    scope = {"time_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
             "scope": "Completed Z/S smoke and 1000-step training, all five scheduled checkpoints, raw train/calibration inputs, inherited pretraining cache, actual source, immutable plan/wrapper exits, gates and analysis",
             "excluded_active": "All active/partial micro and first50 rollouts; require a later complete increment",
             "prior_complete_archive": "/root/autodl-tmp/qvla-repro/backups/091-gpu-b5-final-20261013-v1",
             "prior_local_tar_sha256": "31ae496b9e14ba7573a703c7123cc8b690e5ffbde33c9b99fed965fdee8121ad",
             "checkpoint_scope": "7B checkpoint and frozen profiles remain server-side and bound by the full verified checkpoint/profile identity in resume-content-gate; prior complete archive carries referenced profile/runtime material. No complete local 7B checkpoint claim.",
             "payload_files": len(origins), "payload_bytes": sum(v["bytes"] for v in origins.values()),
             "server_staging": "Hardlinks where possible; compressed tar and verified local copy provide the portable archive. Hardlinks are not an independent physical server replica."}
    (a.archive / "MANIFEST.json").write_text(json.dumps(origins, indent=2) + "\n")
    (a.archive / "SCOPE.json").write_text(json.dumps(scope, indent=2) + "\n")
    files = sorted(p for p in a.archive.rglob("*") if p.is_file())
    sums = {p.relative_to(a.archive).as_posix(): digest(p) for p in files}
    (a.archive / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for name, h in sums.items()))
    tar_path = a.archive / "complete.tar.gz"
    with tarfile.open(tar_path, "w:gz", compresslevel=1) as tar:
        for path in files + [a.archive / "SHA256SUMS.txt"]:
            tar.add(path, arcname=path.relative_to(a.archive).as_posix(), recursive=False)
    with tarfile.open(tar_path, "r:gz") as tar:
        seen = set()
        for member in tar:
            stream = tar.extractfile(member)
            if stream is None or member.name not in sums and member.name != "SHA256SUMS.txt":
                raise ValueError("Unexpected archive member")
            actual = hashlib.sha256()
            for part in iter(lambda: stream.read(4 * 1024**2), b""):
                actual.update(part)
            expected = sums.get(member.name) or digest(a.archive / "SHA256SUMS.txt")
            if actual.hexdigest() != expected:
                raise ValueError(f"Compressed archive member mismatch: {member.name}")
            seen.add(member.name)
        if seen != set(sums) | {"SHA256SUMS.txt"}:
            raise ValueError("Compressed archive incomplete")
    receipt = {"gate": "PASS_COMPLETED_B3_TRAINING_SERVER_ARCHIVE", "archive": str(a.archive),
               "archive_file": str(tar_path), "bytes": tar_path.stat().st_size, "sha256": digest(tar_path),
               "sha_entries": len(sums), "payload_files": len(origins), "all_tar_members_verified": True,
               "completed_wrapper_records": len(completed), "excluded_active_evaluation": True,
               "free_bytes_after": shutil.disk_usage(a.archive).free}
    (a.archive / "SERVER_VERIFICATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
