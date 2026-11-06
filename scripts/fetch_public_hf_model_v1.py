"""Download one pinned public model from Hugging Face via a mirror transport.

Use a commit-pinned snapshot and verify LFS SHA256 or Git blob SHA1 for every
file. The mirror is only a transport; no result is admitted without hashes.
Resume is restricted to an exact byte range and no existing final file is
overwritten. This does not install or execute model code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


ENDPOINT = "https://hf-mirror.com"
ALLOWED = {
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
    "youliangtan/gr00t-n1.5-libero-object-posttrain",
    "youliangtan/gr00t-n1.5-libero-goal-posttrain",
    "youliangtan/gr00t-n1.5-libero-long-posttrain",
}
MAX_BYTES = 20 * 1024**3


def listing(repo):
    if repo not in ALLOWED:
        raise ValueError("Model repository is not in this round's allowlist")
    with urlopen(ENDPOINT + "/api/models/" + repo + "?blobs=true", timeout=60) as response:
        data = json.load(response)
    revision = data.get("sha")
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Unpinned model revision")
    rows = sorted(data["siblings"], key=lambda x: x["rfilename"])
    total = 0
    for row in rows:
        name = row["rfilename"]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or row.get("size") is None:
            raise ValueError("Unexpected file in model snapshot")
        total += int(row["size"])
        if not (row.get("lfs", {}).get("sha256") or row.get("blobId")):
            raise ValueError("No authoritative file hash: " + name)
    if total > MAX_BYTES:
        raise ValueError("Model snapshot exceeds registered transfer budget")
    return revision, rows, total


def verify(path, row):
    size = int(row["size"])
    if path.stat().st_size != size:
        raise ValueError("Wrong byte count: " + str(path))
    sha = hashlib.sha256()
    gitsha = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            sha.update(block)
            gitsha.update(block)
    expected_lfs = row.get("lfs", {}).get("sha256")
    if expected_lfs:
        if sha.hexdigest() != expected_lfs:
            raise ValueError("LFS SHA256 mismatch: " + str(path))
    elif gitsha.hexdigest() != row["blobId"]:
        raise ValueError("Git blob SHA1 mismatch: " + str(path))
    return sha.hexdigest()


def fetch_one(root, repo, revision, row):
    dest = root / row["rfilename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return verify(dest, row)
    partial = dest.with_name(dest.name + ".partial")
    url = ENDPOINT + "/" + repo + "/resolve/" + revision + "/" + quote(row["rfilename"], safe="/")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > int(row["size"]):
            raise ValueError("Partial file exceeds expected size")
        if offset == int(row["size"]):
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if offset and response.status != 206:
                    raise ValueError("Mirror ignored resume range")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resume range")
                with partial.open("ab" if offset else "wb") as output:
                    while True:
                        block = response.read(4 * 1024**2)
                        if not block:
                            break
                        output.write(block)
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))
    sha = verify(partial, row)
    partial.rename(dest)
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    revision, rows, total = listing(args.repo)
    summary = {"repo": args.repo, "revision": revision, "files": len(rows), "bytes": total}
    if args.list_only:
        print(json.dumps(summary))
        return
    root = args.destination.resolve()
    if root.exists() and root.is_symlink():
        raise ValueError("Destination cannot be a symlink")
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < total + 2 * 1024**3:
        raise ValueError("Insufficient disk for model and reserve")
    root.mkdir(parents=True, exist_ok=True)
    receipt = []
    for row in rows:
        sha = fetch_one(root, args.repo, revision, row)
        receipt.append({"path": row["rfilename"], "bytes": int(row["size"]), "sha256": sha,
                        "source_lfs_sha256": row.get("lfs", {}).get("sha256"), "source_git_blob": row.get("blobId")})
        print(json.dumps({"verified": receipt[-1]}), flush=True)
    output = root / "DOWNLOAD_VERIFICATION.json"
    if output.exists():
        raise ValueError("Verification receipt already exists")
    output.write_bytes((json.dumps({**summary, "files_verified": receipt}, indent=2) + "\n").encode())
    print(json.dumps({"gate": "PASS_PUBLIC_MODEL_SNAPSHOT_HASHES", **summary}))


if __name__ == "__main__":
    main()
