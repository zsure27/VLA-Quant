"""Fetch the public pi05_libero checkpoint from its official GCS bucket.

The object list is pinned by GCS generation and every file is checked against
the provider's CRC32C (and MD5 where supplied). No credentials are required.
This tool never deletes or overwrites an existing verified asset.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_libero/"
API = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
MAX_BYTES = 14 * 1024**3


def listing():
    url = API + "?" + urlencode({"prefix": PREFIX, "maxResults": 1000})
    with urlopen(url, timeout=60) as response:
        data = json.load(response)
    if data.get("nextPageToken") or not data.get("items"):
        raise ValueError("Unexpected incomplete GCS object listing")
    rows = sorted(data["items"], key=lambda x: x["name"])
    total = sum(int(row["size"]) for row in rows)
    if total > MAX_BYTES:
        raise ValueError("Public checkpoint exceeds the registered transfer budget")
    for row in rows:
        name = row["name"]
        rel = Path(name.removeprefix(PREFIX))
        if not name.startswith(PREFIX) or rel.is_absolute() or ".." in rel.parts or not row.get("crc32c"):
            raise ValueError("Unexpected object in GCS listing")
    return rows, total


def hashes(path):
    import google_crc32c
    sha, md5, crc = hashlib.sha256(), hashlib.md5(), google_crc32c.Checksum()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            sha.update(block)
            md5.update(block)
            crc.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode(), base64.b64encode(crc.digest()).decode()


def verify(path, row):
    if path.stat().st_size != int(row["size"]):
        raise ValueError("Wrong byte count: " + str(path))
    sha, md5, crc = hashes(path)
    if crc != row["crc32c"] or (row.get("md5Hash") and md5 != row["md5Hash"]):
        raise ValueError("Provider checksum mismatch: " + str(path))
    return sha


def fetch_one(root, row):
    rel = Path(row["name"][len(PREFIX):])
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return verify(dest, row)
    partial = dest.with_name(dest.name + ".partial")
    url = "https://storage.googleapis.com/" + BUCKET + "/" + quote(row["name"], safe="/")
    url += "?" + urlencode({"generation": row["generation"]})
    if int(row["size"]) >= 64 * 1024**2 and shutil.which("aria2c"):
        subprocess.run([
            "aria2c", "--continue=true", "--file-allocation=none",
            "--max-connection-per-server=8", "--split=8", "--min-split-size=4M",
            "--max-tries=5", "--retry-wait=5", "--timeout=90",
            "--summary-interval=30", "--dir=" + str(dest.parent),
            "--out=" + partial.name, url,
        ], check=True)
        sha = verify(partial, row)
        partial.rename(dest)
        return sha
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > int(row["size"]):
            raise ValueError("Partial file exceeds object size")
        if offset == int(row["size"]):
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if offset and response.status != 206:
                    raise ValueError("Server ignored resume range")
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
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    rows, total = listing()
    summary = {"source": "gs://" + BUCKET + "/" + PREFIX, "objects": len(rows), "bytes": total}
    if args.list_only:
        print(json.dumps(summary))
        return
    root = args.destination.resolve()
    if root.exists() and root.is_symlink():
        raise ValueError("Destination cannot be a symlink")
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < total + 2 * 1024**3:
        raise ValueError("Insufficient disk for checkpoint and reserve")
    root.mkdir(parents=True, exist_ok=True)
    receipt = []
    for row in rows:
        sha = fetch_one(root, row)
        receipt.append({"path": row["name"][len(PREFIX):], "bytes": int(row["size"]),
                        "generation": row["generation"], "sha256": sha, "gcs_crc32c": row["crc32c"]})
        print(json.dumps({"verified": receipt[-1]}), flush=True)
    output = root / "DOWNLOAD_VERIFICATION.json"
    if output.exists():
        raise ValueError("Verification receipt already exists; preserve its identity")
    output.write_bytes((json.dumps({**summary, "files": receipt}, indent=2) + "\n").encode())
    print(json.dumps({"gate": "PASS_PUBLIC_GCS_CHECKPOINT_SHA_AND_CRC32C", **summary}))


if __name__ == "__main__":
    main()
