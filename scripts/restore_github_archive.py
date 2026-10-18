"""Restore one verified archive from the private VLA GitHub Release store.

The manifest is the JSON receipt produced by github_archive_store.py. Listing
is offline. Restoring creates a new file, verifies every part and the complete
archive, and never extracts it or replaces an existing file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY = "zsure27/VLA-Quant-Archives"
API_ROOT = "https://api.github.com/repos/" + REPOSITORY
SIGNED_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}
BUFFER_BYTES = 4 * 1024**2
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_record(key, record):
    """Validate the receipt without trusting paths or URLs embedded in it."""
    require(isinstance(key, str) and SHA256_PATTERN.fullmatch(key), "Invalid archive SHA256")
    require(isinstance(record, dict) and record.get("sha256") == key, "Archive SHA256/key mismatch")
    require(positive_integer(record.get("bytes")), "Invalid archive length")
    require(record.get("remote_download_sha256_verified") is True, "Archive lacks remote verification")
    parts = record.get("parts")
    require(isinstance(parts, list) and parts, "Archive has no parts")
    offset = 0
    asset_ids = set()
    for index, part in enumerate(parts, 1):
        require(isinstance(part, dict), "Invalid part record")
        require(positive_integer(part.get("asset_id")), "Invalid asset ID")
        require(part["asset_id"] not in asset_ids, "Repeated asset ID")
        asset_ids.add(part["asset_id"])
        require(positive_integer(part.get("bytes")), "Invalid part length")
        require(type(part.get("offset")) is int and part["offset"] == offset, "Part offsets are not contiguous")
        digest = part.get("sha256")
        require(isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest), "Invalid part SHA256")
        expected_name = "sha256-" + digest + (".part%03d-of%03d" % (index, len(parts)))
        require(part.get("name") == expected_name, "Part name/order mismatch")
        require(part.get("download_verified") is True, "Part lacks remote verification")
        offset += part["bytes"]
    require(offset == record["bytes"], "Part lengths do not equal archive length")
    return record


def read_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(data, dict) and data.get("repository") == REPOSITORY,
            "Manifest must name the private repository " + REPOSITORY)
    files = data.get("files")
    require(isinstance(files, dict), "Manifest has no files mapping")
    for key, record in files.items():
        validate_record(key, record)
    return files


def credential_token():
    environment = os.environ.copy()
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\nusername=zsure27\n\n",
        text=True, capture_output=True, env=environment, check=False,
    )
    # Credential output and stderr must never be included in errors or receipts.
    require(result.returncode == 0, "GitHub credentials are unavailable in Git Credential Manager")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    require(bool(token), "Git Credential Manager returned no GitHub token")
    return token


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def signed_url_allowed(url):
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.hostname in SIGNED_HOSTS
            and port in (None, 443) and parsed.username is None and parsed.password is None)


class GitHubAssets:
    def __init__(self, token):
        self.opener = urllib.request.build_opener(NoRedirect())
        self.headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "User-Agent": "VLA-archive-restore",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def verify_private_repository(self):
        request = urllib.request.Request(API_ROOT, headers=self.headers)
        try:
            with self.opener.open(request, timeout=90) as response:
                metadata = json.loads(response.read(1024**2))
        except urllib.error.HTTPError as error:
            raise RuntimeError("GitHub repository check failed, HTTP " + str(error.code)) from None
        require(metadata.get("private") is True and metadata.get("full_name") == REPOSITORY,
                "Restore destination repository is not the expected private repository")

    def open_part(self, part):
        # Never use a manifest-supplied URL for an authenticated request.
        url = API_ROOT + "/releases/assets/" + str(part["asset_id"])
        request = urllib.request.Request(url, headers={**self.headers, "Accept": "application/octet-stream"})
        for attempt in range(5):
            try:
                return self.opener.open(request, timeout=180)
            except urllib.error.HTTPError as error:
                if error.code not in (301, 302, 303, 307, 308):
                    raise RuntimeError("GitHub asset download failed, HTTP " + str(error.code)) from None
                location = error.headers.get("Location", "")
                error.close()
                require(signed_url_allowed(location), "Rejected asset redirect outside approved HTTPS hosts")
                # The GitHub bearer token is intentionally absent after redirect.
                request = urllib.request.Request(location, headers={"User-Agent": "VLA-archive-restore"})
        raise RuntimeError("Too many asset redirects")


def publish_without_replace(partial, destination):
    """Publish the complete file atomically, with no overwrite race."""
    if os.name == "nt":
        # Windows rename fails if the destination already exists.
        os.rename(partial, destination)
    else:
        # link is an atomic no-replace publication on POSIX; remove our own temp.
        os.link(partial, destination)
        os.unlink(partial)


def restore_record(record, destination, open_part, progress=None):
    """Stream an archive; injectable input streams allow offline verification."""
    validate_record(record.get("sha256"), record)
    destination = Path(os.path.abspath(os.fspath(destination)))
    partial = destination.with_name(destination.name + ".partial")
    require(destination.parent.is_dir(), "Destination parent directory must already exist")
    require(not os.path.lexists(destination), "Destination already exists; choose a new file path")
    require(not os.path.lexists(partial), "Partial file already exists; choose a new file path")
    require(shutil.disk_usage(destination.parent).free >= record["bytes"], "Insufficient destination disk space")
    whole_hash = hashlib.sha256()
    whole_bytes = 0
    # Exclusive creation prevents replacing an existing or concurrently created file.
    with partial.open("xb") as output:
        for index, part in enumerate(record["parts"], 1):
            part_hash = hashlib.sha256()
            part_bytes = 0
            last_progress = time.monotonic()
            with open_part(part) as response:
                while True:
                    chunk = response.read(BUFFER_BYTES)
                    if not chunk:
                        break
                    require(part_bytes + len(chunk) <= part["bytes"], "Downloaded part exceeds expected length")
                    output.write(chunk)
                    part_hash.update(chunk)
                    whole_hash.update(chunk)
                    part_bytes += len(chunk)
                    whole_bytes += len(chunk)
                    if progress and time.monotonic() - last_progress >= 45:
                        progress({"part": index, "parts": len(record["parts"]),
                                  "received_bytes": whole_bytes, "total_bytes": record["bytes"]})
                        last_progress = time.monotonic()
            require(part_bytes == part["bytes"], "Downloaded part length mismatch")
            require(part_hash.hexdigest() == part["sha256"], "Downloaded part SHA256 mismatch")
            if progress:
                progress({"verified_part": index, "parts": len(record["parts"])})
        require(whole_bytes == record["bytes"], "Archive length mismatch")
        require(whole_hash.hexdigest() == record["sha256"], "Archive SHA256 mismatch")
        output.flush()
        os.fsync(output.fileno())
    publish_without_replace(partial, destination)
    return {"restored": str(destination), "bytes": whole_bytes,
            "sha256": whole_hash.hexdigest(), "parts_verified": len(record["parts"]),
            "extracted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="List verified archives offline")
    parser.add_argument("--sha256", help="SHA256 of the archive to restore")
    parser.add_argument("--destination", type=Path, help="Exact new output file path; parent must exist")
    args = parser.parse_args(argv)
    files = read_manifest(args.manifest)
    if args.list:
        require(args.sha256 is None and args.destination is None, "Use --list without restore options")
        for digest, record in sorted(files.items()):
            print(json.dumps({"sha256": digest, "source_name": record.get("source_name", ""),
                              "source_path": record.get("source_path", ""),
                              "bytes": record["bytes"], "parts": len(record["parts"])}, ensure_ascii=True))
        return
    require(args.sha256 is not None and args.destination is not None,
            "Restore requires --sha256 and --destination; use --list for offline inventory")
    require(args.sha256 in files, "Selected SHA256 is absent from the verified manifest")
    destination = Path(os.path.abspath(os.fspath(args.destination)))
    require(destination.parent.is_dir(), "Destination parent directory must already exist")
    require(not os.path.lexists(destination) and not os.path.lexists(str(destination) + ".partial"),
            "Destination or partial file already exists; choose a new file path")
    client = GitHubAssets(credential_token())
    client.verify_private_repository()
    result = restore_record(files[args.sha256], destination, client.open_part,
                            progress=lambda event: print(json.dumps(event), flush=True))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        # Do not print URLs, signed query parameters, credentials or HTTP bodies.
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
        print("Restore failed: " + message + ". Any created .partial file is retained; it is not a verified archive.", file=sys.stderr)
        sys.exit(1)
