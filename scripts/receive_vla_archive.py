"""Stream a verified remote VLA closure directory to local storage over SSH.

This avoids one SFTP round trip per file when an archive contains many traces.
The caller must run verify_vla_archive_local.py after this transfer completes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    name = args.archive_name
    if not name.startswith("b2b3-046-final-") or not name.replace("-", "").isalnum():
        raise ValueError("Unexpected archive name")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = ["ssh", "-i", str(args.identity), "-p", str(args.port),
               "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", f"UserKnownHostsFile={args.known_hosts}", args.host,
               f"tar -C /root/autodl-tmp/qvla-repro/backups -cf - {name}"]
    count = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    target = (destination / member.name).resolve()
                    if destination not in target.parents:
                        raise ValueError(f"Unsafe archive path: {member.name}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError(f"Cannot extract: {member.name}")
                        with source, target.open("wb") as output:
                            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                        count += 1
                    else:
                        raise ValueError(f"Unsupported archive member: {member.name}")
            error = process.stderr.read().decode("utf-8", errors="replace")
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"Remote tar transfer failed ({code}): {error}")
        except Exception:
            process.kill()
            process.wait()
            raise
    print(f"TRANSFERRED_FILES={count}")


if __name__ == "__main__":
    main()
