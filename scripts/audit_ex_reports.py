"""Verify moved evidence against original working-copy or portable Git blob identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def audit(staged=False):
    manifest = json.loads((ROOT / "Ex_Reports/migration-manifest.json").read_text(encoding="utf-8"))
    verified, local_absent = 0, 0
    for row in manifest["files"]:
        path = ROOT / row["new_path"]
        if row.get("storage") in ("local_only_ignored", "private_git") and not path.exists():
            local_absent += 1
            continue
        if path.is_symlink() or ROOT.resolve() not in path.resolve().parents:
            raise ValueError("Unsafe relocated evidence")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual not in (row["sha256"], row.get("git_blob_sha256")):
            raise ValueError("Evidence changed: " + row["new_path"])
        verified += 1
    if staged:
        raw = subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT), "ls-files", "--stage", "-z"])
        blobs = {entry.split("\t", 1)[1]: entry.split()[1] for entry in raw.decode().split("\0") if entry}
        for row in manifest["files"]:
            if "git_blob_sha1" in row and blobs.get(row["new_path"]) != row["git_blob_sha1"]:
                raise ValueError("Portable Git blob changed: " + row["new_path"])
    return {"gate": "PASS_EX_REPORTS_RELOCATION", "files_verified": verified,
            "ignored_local_files_absent_from_checkout": local_absent,
            "historical_tasks": len({r["task"] for r in manifest["files"]}), "git_blob_identity_checked": staged}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staged", action="store_true")
    print(json.dumps(audit(p.parse_args().staged)))
