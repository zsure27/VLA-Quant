"""Bounded parallel, resumable archive storage in the approved private GitHub repo.

Never deletes local or remote files. Each part is checkpointed only after a full
remote download matches its SHA256. Do not share a receipt with another process.
Existing v1 receipts and content-addressed asset names are supported. A receipt
is a trust record: retain it in the approved audit directory, not from strangers.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = "zsure27/VLA-Quant-Archives"
TAG = "backup-archive-20261017"
CHUNK = 1024 ** 3
READ_SIZE = 1024 ** 2
DOWNLOAD_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PRINT_LOCK = threading.Lock()
MAX_WORKERS = 16


class ArchiveError(RuntimeError):
    pass


class HashMismatch(ArchiveError):
    pass


class RateLimitStop(ArchiveError):
    pass


class HTTPFailure(ArchiveError):
    def __init__(self, status, headers=None):
        self.status = status
        self.headers = headers or {}
        super().__init__("GitHub HTTP " + str(status))


def report(**event):
    with PRINT_LOCK:
        print(json.dumps(event, ensure_ascii=False), flush=True)


def retryable(error):
    if isinstance(error, (HashMismatch, RateLimitStop)):
        return False
    if isinstance(error, (HTTPFailure, urllib.error.HTTPError)):
        return error.status in (408, 429, 500, 502, 503, 504) if isinstance(error, HTTPFailure) else error.code in (408, 429, 500, 502, 503, 504)
    return isinstance(error, (urllib.error.URLError, TimeoutError, socket.timeout,
                              ConnectionError, http.client.HTTPException))


def delay_for(error, attempt):
    """Bounded retries: never silently wait until an exhausted rate limit resets."""
    retry_after = getattr(error, "headers", {}).get("Retry-After")
    try:
        retry_after = float(retry_after)
    except (TypeError, ValueError):
        retry_after = 0.0
    if retry_after > 30:
        raise RateLimitStop("GitHub requests a pause over 30 seconds; resume later")
    return max(retry_after, min(2 ** attempt, 16))


def with_retries(operation, attempts=5, sleeper=time.sleep):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if not retryable(error) or attempt + 1 == attempts:
                raise
            sleeper(delay_for(error, attempt))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        return None


def safe_url(url, hosts, path_prefix=None):
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in hosts or
            parts.username or parts.password or parts.port not in (None, 443) or
            (path_prefix is not None and not parts.path.startswith(path_prefix))):
        raise ArchiveError("Unexpected GitHub endpoint or redirect host")
    return parts


class RequestBudget:
    """Share API pacing and reserve 100 primary requests for recovery/status."""
    def __init__(self, interval=0.25, reserve=100):
        self.interval = interval
        self.reserve = reserve
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.remaining = None

    def acquire(self):
        with self.lock:
            if self.remaining is not None and self.remaining <= self.reserve:
                raise RateLimitStop("GitHub API reserve reached; resume later")
            delay = self.next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_request = time.monotonic() + self.interval
            if self.remaining is not None:
                self.remaining -= 1

    def observe(self, headers):
        value = headers.get("X-RateLimit-Remaining")
        if value is not None:
            with self.lock:
                # Concurrent responses may arrive out of order; use the lower bound.
                value = int(value)
                self.remaining = value if self.remaining is None else min(self.remaining, value)


@dataclass(frozen=True)
class Part:
    whole_sha256: str
    source: Path
    file_bytes: int
    source_mtime_ns: int
    index: int
    count: int
    offset: int
    size: int
    sha256: str

    @property
    def name(self):
        return "sha256-" + self.sha256 + ".part%03d-of%03d" % (self.index + 1, self.count)


def prepare_parts(plan, chunk_size=CHUNK):
    if plan.get("repository") != REPO or plan.get("security_scan_passed") is not True:
        raise ArchiveError("Plan must name the approved private repo and pass its security scan")
    if "tag" in plan and plan["tag"] != TAG:
        raise ArchiveError("Unexpected release tag")
    if chunk_size < 1 or not isinstance(plan.get("files"), list):
        raise ArchiveError("Invalid plan")
    all_parts = []
    seen = {}
    for entry in plan["files"]:
        key, size = entry["sha256"], entry["bytes"]
        if not HEX_SHA256.fullmatch(key) or type(size) is not int or size <= 0:
            raise ArchiveError("Invalid SHA256 or archive length")
        if key in seen:
            if seen[key] != size:
                raise ArchiveError("Conflicting duplicate archive lengths")
            continue
        source = Path(entry["path"]).resolve(strict=True)
        before = source.stat()
        if not source.is_file() or before.st_size != size:
            raise ArchiveError("Source size mismatch: " + source.name)
        count = (size + chunk_size - 1) // chunk_size
        parts = []
        whole = hashlib.sha256()
        with source.open("rb") as stream:
            for index in range(count):
                offset = index * chunk_size
                length = min(chunk_size, size - offset)
                remaining = length
                digest = hashlib.sha256()
                while remaining:
                    block = stream.read(min(READ_SIZE, remaining))
                    if not block:
                        raise ArchiveError("Source truncated during admission")
                    whole.update(block)
                    digest.update(block)
                    remaining -= len(block)
                parts.append(Part(key, source, size, before.st_mtime_ns, index,
                                  count, offset, length, digest.hexdigest()))
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ArchiveError("Source changed during admission: " + source.name)
        if whole.hexdigest() != key:
            raise HashMismatch("Whole source SHA256 differs from plan: " + source.name)
        all_parts.extend(parts)
        seen[key] = size
        report(source_admitted=source.name, bytes=size, parts=count)
    return all_parts


def source_unchanged(part):
    stat = part.source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (part.file_bytes, part.source_mtime_ns):
        raise ArchiveError("Source changed after admission: " + part.source.name)


def validate_asset(asset, part):
    if (asset.get("name") != part.name or asset.get("size") != part.size or
            asset.get("state") != "uploaded" or type(asset.get("id")) is not int):
        raise ArchiveError("Existing asset is incomplete or has conflicting metadata: " + part.name)
    if asset.get("digest") not in (None, "sha256:" + part.sha256):
        raise HashMismatch("Remote asset digest mismatch: " + part.name)
    safe_url(asset["browser_download_url"], {"github.com"},
             "/" + REPO + "/releases/download/" + TAG + "/")


class Receipt:
    def __init__(self, path, release):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "repository": REPO, "release": release["html_url"],
            "release_id": release["id"], "tag": TAG, "files": {}}
        if (self.state.get("repository") != REPO or self.state.get("tag") != TAG or
                self.state.get("release_id") != release["id"] or
                not isinstance(self.state.get("files"), dict)):
            raise ArchiveError("Receipt belongs to a different release")

    def _save(self):
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.state, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def verified(self, part, asset):
        with self.lock:
            file = self.state["files"].get(part.whole_sha256, {})
            if file.get("bytes") != part.file_bytes:
                return False
            return any(p.get("asset_id") == asset["id"] and
                       p.get("name") == part.name and p.get("offset") == part.offset and
                       p.get("bytes") == part.size and p.get("sha256") == part.sha256 and
                       p.get("download_verified") is True for p in file.get("parts", []))

    def record(self, part, asset):
        with self.lock:
            file = self.state["files"].setdefault(part.whole_sha256, {
                "source_name": part.source.name, "source_path": str(part.source),
                "bytes": part.file_bytes, "sha256": part.whole_sha256, "parts": [],
                "remote_download_sha256_verified": False})
            if file.get("bytes") != part.file_bytes or file.get("sha256") != part.whole_sha256:
                raise ArchiveError("Conflicting file metadata in receipt")
            file["parts"] = [p for p in file.get("parts", []) if p.get("offset") != part.offset]
            file["parts"].append({"asset_id": asset["id"], "name": part.name,
                                  "url": asset["browser_download_url"],
                                  "offset": part.offset, "bytes": part.size,
                                  "sha256": part.sha256, "download_verified": True})
            file["parts"].sort(key=lambda p: p["offset"])
            # Preserve v1 compatibility: only verified_complete sets whole completion.
            file["remote_download_sha256_verified"] = False
            self._save()

    def verified_complete(self, parts):
        by_file = {}
        for part in parts:
            by_file.setdefault(part.whole_sha256, []).append(part)
        with self.lock:
            for key, expected in by_file.items():
                file = self.state["files"].get(key)
                if file is None or len(file.get("parts", [])) != len(expected):
                    raise ArchiveError("Archive receipt lacks required parts")
                for part in expected:
                    source_unchanged(part)
                    p = next((p for p in file["parts"] if p.get("offset") == part.offset), {})
                    if (p.get("bytes"), p.get("sha256"), p.get("name"),
                            p.get("download_verified")) != (part.size, part.sha256, part.name, True):
                        raise ArchiveError("Archive receipt contains an unverified part")
                file["remote_download_sha256_verified"] = True
            self._save()


class GitHub:
    def __init__(self, token):
        self.headers = {"Authorization": "Bearer " + token,
                        "Accept": "application/vnd.github+json",
                        "User-Agent": "VLA-archive-store-v2",
                        "X-GitHub-Api-Version": "2022-11-28"}
        self.budget = RequestBudget()
        self.asset_lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.name_locks = {}
        self.assets = {}
        if self.api("").get("private") is not True:
            raise ArchiveError("Archive destination must remain private")
        self.release = self.api("releases/tags/" + TAG)
        if self.release.get("tag_name") != TAG:
            raise ArchiveError("Unexpected release identity")
        self.refresh_assets()

    def api(self, path):
        url = "https://api.github.com/repos/" + REPO + ("/" + path if path else "")
        safe_url(url, {"api.github.com"}, "/repos/" + REPO)
        def request():
            self.budget.acquire()
            try:
                with urllib.request.build_opener(NoRedirect()).open(
                        urllib.request.Request(url, headers=self.headers), timeout=90) as response:
                    self.budget.observe(response.headers)
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                self.budget.observe(error.headers)
                if error.code == 403 and (error.headers.get("Retry-After") or
                        error.headers.get("X-RateLimit-Remaining") == "0"):
                    raise RateLimitStop("GitHub rate limit reached; resume later") from None
                raise
        return with_retries(request)

    def refresh_assets(self):
        with self.refresh_lock:
            fresh = {}
            page = 1
            while True:
                batch = self.api("releases/%s/assets?per_page=100&page=%s" % (self.release["id"], page))
                for asset in batch:
                    fresh[asset["name"]] = asset
                if len(batch) < 100:
                    break
                page += 1
                if page > 100:
                    raise ArchiveError("Unexpected asset listing size")
            with self.asset_lock:
                self.assets.update(fresh)

    def get_asset(self, name):
        with self.asset_lock:
            return self.assets.get(name)

    def name_lock(self, name):
        with self.asset_lock:
            return self.name_locks.setdefault(name, threading.Lock())

    def upload(self, part):
        source_unchanged(part)
        url = self.release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(part.name)
        parsed = safe_url(url, {"uploads.github.com"}, "/repos/" + REPO + "/releases/%s/assets" % self.release["id"])
        self.budget.acquire()
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=180)
        try:
            connection.putrequest("POST", parsed.path + "?" + parsed.query)
            for name, value in dict(self.headers, **{"Content-Type": "application/octet-stream",
                                                    "Content-Length": str(part.size)}).items():
                connection.putheader(name, value)
            connection.endheaders()
            sent = 0
            digest = hashlib.sha256()
            last = time.monotonic()
            report(upload=part.source.name, part=part.index + 1, of=part.count, bytes=part.size)
            with part.source.open("rb") as stream:
                stream.seek(part.offset)
                while sent < part.size:
                    block = stream.read(min(READ_SIZE, part.size - sent))
                    if not block:
                        raise ArchiveError("Source truncated during upload")
                    digest.update(block)
                    connection.send(block)
                    sent += len(block)
                    if time.monotonic() - last >= 45:
                        report(asset=part.name, part_progress_percent=round(100 * sent / part.size, 1))
                        last = time.monotonic()
            if digest.hexdigest() != part.sha256:
                raise HashMismatch("Source part changed during upload")
            response = connection.getresponse()
            self.budget.observe(response.headers)
            raw = response.read(1024 ** 2)
            if response.status != 201:
                raise HTTPFailure(response.status, response.headers)
            asset = json.loads(raw)
            with self.asset_lock:
                self.assets[part.name] = asset
            return asset
        finally:
            connection.close()

    def ensure_asset(self, part):
        # The per-name lock also de-duplicates identical parts from separate files.
        for attempt in range(5):
            asset = self.get_asset(part.name)
            if asset is not None and asset.get("state") == "uploaded":
                validate_asset(asset, part)
                return asset
            try:
                if asset is None:
                    # Another independent uploader may have created this asset
                    # since our startup listing. Refresh before sending its body.
                    self.refresh_assets()
                    asset = self.get_asset(part.name)
                    if asset is not None:
                        if asset.get("state") == "uploaded":
                            validate_asset(asset, part)
                            return asset
                        raise HTTPFailure(503)
                    asset = self.upload(part)
                    validate_asset(asset, part)
                    return asset
                raise HTTPFailure(503)  # eventual starter asset; never delete/overwrite it
            except Exception as error:
                conflict = isinstance(error, HTTPFailure) and error.status in (409, 422)
                if not conflict and not retryable(error):
                    raise
                # An upload may succeed despite a lost response: list before retrying.
                self.refresh_assets()
                asset = self.get_asset(part.name)
                if asset is not None and asset.get("state") == "uploaded":
                    validate_asset(asset, part)
                    return asset
                if attempt == 4:
                    raise ArchiveError("Upload unresolved after retries; resume later: " + part.name) from error
                time.sleep(delay_for(error, attempt))

    def open_download(self, asset):
        # Authorization goes only to the fixed API endpoint, never a signed CDN URL.
        url = "https://api.github.com/repos/" + REPO + "/releases/assets/" + str(asset["id"])
        headers = dict(self.headers, Accept="application/octet-stream")
        self.budget.acquire()
        opener = urllib.request.build_opener(NoRedirect())
        for redirects in range(4):
            try:
                response = opener.open(urllib.request.Request(url, headers=headers), timeout=180)
                if redirects == 0:
                    self.budget.observe(response.headers)
                return response
            except urllib.error.HTTPError as error:
                if redirects == 0:
                    self.budget.observe(error.headers)
                if error.code not in (301, 302, 303, 307, 308):
                    raise
                url = error.headers.get("Location", "")
                safe_url(url, DOWNLOAD_HOSTS)
                headers = {"User-Agent": "VLA-archive-restore-check-v2"}
        raise ArchiveError("Too many download redirects")

    def verify_download(self, asset, part):
        def verify():
            digest = hashlib.sha256()
            received = 0
            with self.open_download(asset) as response:
                while True:
                    block = response.read(READ_SIZE)
                    if not block:
                        break
                    received += len(block)
                    if received > part.size:
                        raise HashMismatch("Remote part exceeds expected length")
                    digest.update(block)
            if received != part.size or digest.hexdigest() != part.sha256:
                raise HashMismatch("Remote download SHA256 mismatch: " + part.name)
        with_retries(verify)


def run_parts(parts, client, receipt, workers=4):
    if not 1 <= workers <= MAX_WORKERS:
        raise ArchiveError("Workers must be between 1 and " + str(MAX_WORKERS))
    by_file = {}
    for part in parts:
        by_file.setdefault(part.whole_sha256, []).append(part)

    def complete_file_if_ready(part):
        # Completion is durable per archive, so a later archive failure cannot
        # hide an already recoverable file. Hold the receipt's RLock across the
        # readiness check and full part/source validation to avoid racing peers.
        with receipt.lock:
            file = receipt.state["files"].get(part.whole_sha256)
            expected = by_file[part.whole_sha256]
            if file is None or len(file.get("parts", [])) != len(expected):
                return
            was_complete = file.get("remote_download_sha256_verified") is True
            receipt.verified_complete(expected)
            if not was_complete:
                report(verified_archive=part.source.name, bytes=part.file_bytes,
                       sha256=part.whole_sha256)

    def one(part):
        source_unchanged(part)
        with client.name_lock(part.name):
            asset = client.ensure_asset(part)
            validate_asset(asset, part)
            if receipt.verified(part, asset):
                report(reused_verified_part=part.name, file=part.source.name)
                complete_file_if_ready(part)
                return
            client.verify_download(asset, part)
            source_unchanged(part)
            receipt.record(part, asset)
            report(verified_part=part.index + 1, of=part.count, file=part.source.name)
            complete_file_if_ready(part)
    # At most `workers` submitted tasks; failure does not start the remaining plan.
    iterator = iter(parts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, p) for p in [next(iterator, None) for _ in range(workers)] if p is not None}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            errors = []
            for future in done:
                try:
                    future.result()
                except Exception as error:
                    errors.append(error)
            if errors:
                for future in pending:
                    future.cancel()
                # Already active parts may finish and checkpoint before shutdown.
                raise errors[0]
            for _ in done:
                part = next(iterator, None)
                if part is not None:
                    pending.add(pool.submit(one, part))
    receipt.verified_complete(parts)


def credential_token():
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\nusername=zsure27\n\n",
                            text=True, capture_output=True, check=True)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not fields.get("password"):
        raise ArchiveError("Git Credential Manager did not provide a token")
    return fields["password"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, MAX_WORKERS + 1))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    parts = prepare_parts(plan)
    client = GitHub(credential_token())
    receipt = Receipt(args.receipt, client.release)
    run_parts(parts, client, receipt, args.workers)
    report(complete=True, archives=len({p.whole_sha256 for p in parts}),
           parts=len(parts), release=client.release["html_url"], workers=args.workers)


if __name__ == "__main__":
    main()
