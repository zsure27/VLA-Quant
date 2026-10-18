"""Offline recovery, integrity, concurrency, and credential-boundary tests."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import github_archive_store_v2 as archive


RELEASE = {"id": 123, "html_url": "https://github.com/" + archive.REPO +
           "/releases/tag/" + archive.TAG}


class FakeGitHub:
    def __init__(self):
        self.assets = {}
        self.locks = {}
        self.lock = threading.Lock()
        self.uploads = []
        self.downloads = []
        self.fail_name = None
        self.corrupt = False
        self.latency = 0
        self.active = 0
        self.max_active = 0

    def name_lock(self, name):
        with self.lock:
            return self.locks.setdefault(name, threading.Lock())

    def ensure_asset(self, part):
        with self.lock:
            if part.name not in self.assets:
                self.uploads.append(part.name)
                self.assets[part.name] = {
                    "id": len(self.assets) + 1, "name": part.name,
                    "state": "uploaded", "size": part.size,
                    "digest": "sha256:" + part.sha256,
                    "browser_download_url": "https://github.com/" + archive.REPO +
                    "/releases/download/" + archive.TAG + "/" + part.name}
            return self.assets[part.name]

    def verify_download(self, asset, part):
        with self.lock:
            self.downloads.append(part.name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.latency)
            if part.name == self.fail_name:
                raise ConnectionError("simulated interruption")
            with part.source.open("rb") as stream:
                stream.seek(part.offset)
                remote = stream.read(part.size)
            if self.corrupt:
                remote = b"X" * len(remote)
            if hashlib.sha256(remote).hexdigest() != part.sha256:
                raise archive.HashMismatch("simulated corrupt remote bytes")
        finally:
            with self.lock:
                self.active -= 1


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def parts(self, contents=(b"abcdefghijkl",)):
        files = []
        for index, content in enumerate(contents):
            source = self.root / (str(index) + ".tar.gz")
            source.write_bytes(content)
            files.append({"path": str(source), "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()})
        return archive.prepare_parts({"repository": archive.REPO,
                                      "security_scan_passed": True, "files": files}, chunk_size=4)

    def test_resume_keeps_verified_part_and_reuses_uploaded_uncheckpointed_asset(self):
        parts = self.parts()
        client = FakeGitHub()
        receipt_path = self.root / "receipt.json"
        receipt = archive.Receipt(receipt_path, RELEASE)
        client.fail_name = parts[1].name
        with self.assertRaises(ConnectionError):
            archive.run_parts(parts, client, receipt, workers=1)
        saved = json.loads(receipt_path.read_text())
        file = saved["files"][parts[0].whole_sha256]
        self.assertEqual(len(file["parts"]), 1)
        self.assertFalse(file["remote_download_sha256_verified"])
        self.assertEqual(client.uploads, [parts[0].name, parts[1].name])
        client.fail_name = None
        resumed = archive.Receipt(receipt_path, RELEASE)
        archive.run_parts(parts, client, resumed, workers=1)
        self.assertEqual(client.downloads.count(parts[0].name), 1)
        self.assertEqual(client.uploads.count(parts[1].name), 1)
        final = json.loads(receipt_path.read_text())["files"][parts[0].whole_sha256]
        self.assertTrue(final["remote_download_sha256_verified"])
        self.assertEqual([p["offset"] for p in final["parts"]], [0, 4, 8])

    def test_wrong_whole_source_sha_rejected_before_network(self):
        source = self.root / "source.tar.gz"
        source.write_bytes(b"unchanged local original")
        with self.assertRaises(archive.HashMismatch):
            archive.prepare_parts({"repository": archive.REPO, "security_scan_passed": True,
                                   "files": [{"path": str(source), "bytes": source.stat().st_size,
                                              "sha256": "0" * 64}]}, chunk_size=4)
        self.assertEqual(source.read_bytes(), b"unchanged local original")

    def test_completed_archive_remains_complete_when_later_archive_fails_and_resumes(self):
        parts = self.parts((b"abcdefghijkl", b"mnopqrstuvwx"))
        client = FakeGitHub()
        receipt_path = self.root / "receipt.json"
        receipt = archive.Receipt(receipt_path, RELEASE)
        client.fail_name = parts[4].name
        with self.assertRaises(ConnectionError):
            archive.run_parts(parts, client, receipt, workers=1)
        saved = json.loads(receipt_path.read_text())
        first = saved["files"][parts[0].whole_sha256]
        second = saved["files"][parts[3].whole_sha256]
        self.assertTrue(first["remote_download_sha256_verified"])
        self.assertEqual(len(first["parts"]), 3)
        self.assertFalse(second["remote_download_sha256_verified"])
        self.assertEqual(len(second["parts"]), 1)

        client.fail_name = None
        resumed = archive.Receipt(receipt_path, RELEASE)
        archive.run_parts(parts, client, resumed, workers=1)
        final = json.loads(receipt_path.read_text())
        self.assertTrue(all(file["remote_download_sha256_verified"]
                            for file in final["files"].values()))
        # The first archive stays usable without a second upload or download.
        for part in parts[:3]:
            self.assertEqual(client.uploads.count(part.name), 1)
            self.assertEqual(client.downloads.count(part.name), 1)

    def test_remote_hash_mismatch_never_checkpoints_or_completes(self):
        parts = self.parts((b"abcd",))
        client = FakeGitHub()
        client.corrupt = True
        receipt = archive.Receipt(self.root / "receipt.json", RELEASE)
        with self.assertRaises(archive.HashMismatch):
            archive.run_parts(parts, client, receipt, workers=1)
        self.assertFalse(receipt.state["files"])
        self.assertEqual(parts[0].source.read_bytes(), b"abcd")

    def test_parallelism_bounded_and_supports_parts_from_multiple_archives(self):
        parts = self.parts((b"abcdefghijkl", b"mnopqrstuvwx"))
        client = FakeGitHub()
        client.latency = .04
        receipt = archive.Receipt(self.root / "receipt.json", RELEASE)
        archive.run_parts(parts, client, receipt, workers=4)
        self.assertGreaterEqual(client.max_active, 2)
        self.assertLessEqual(client.max_active, 4)
        self.assertEqual(len(receipt.state["files"]), 2)
        self.assertTrue(all(f["remote_download_sha256_verified"] for f in receipt.state["files"].values()))

    def test_existing_asset_replacement_forces_download_again(self):
        parts = self.parts((b"abcd",))
        client = FakeGitHub()
        receipt = archive.Receipt(self.root / "receipt.json", RELEASE)
        archive.run_parts(parts, client, receipt, workers=1)
        client.assets[parts[0].name]["id"] = 99
        archive.run_parts(parts, client, receipt, workers=1)
        self.assertEqual(client.downloads.count(parts[0].name), 2)

    def test_stale_client_refreshes_and_reuses_other_uploader_asset_before_post(self):
        part = self.parts((b"abcd",))[0]
        asset = FakeGitHub().ensure_asset(part)
        client = archive.GitHub.__new__(archive.GitHub)
        with patch.object(client, 'get_asset', side_effect=[None, asset]), \
                patch.object(client, 'refresh_assets') as refresh, \
                patch.object(client, 'upload') as upload:
            self.assertEqual(client.ensure_asset(part), asset)
        refresh.assert_called_once_with()
        upload.assert_not_called()

    def test_transient_retries_bounded_and_hash_failure_is_not_retried(self):
        attempts = []
        pauses = []
        def transient():
            attempts.append(1)
            if len(attempts) < 3:
                raise archive.HTTPFailure(503)
            return "ok"
        self.assertEqual(archive.with_retries(transient, sleeper=pauses.append), "ok")
        self.assertEqual(pauses, [1, 2])
        attempts.clear()
        def corrupt():
            attempts.append(1)
            raise archive.HashMismatch("reject")
        with self.assertRaises(archive.HashMismatch):
            archive.with_retries(corrupt, sleeper=pauses.append)
        self.assertEqual(len(attempts), 1)

    def test_token_not_forwarded_to_cdn_and_unknown_redirect_rejected(self):
        client = archive.GitHub.__new__(archive.GitHub)
        client.headers = {"Authorization": "Bearer test-only-token"}
        client.budget = archive.RequestBudget(interval=0)
        calls = []
        class Opener:
            def open(self, req, timeout):
                calls.append(req)
                if len(calls) == 1:
                    raise urllib.error.HTTPError(req.full_url, 302, "redirect",
                                                 {"Location": "https://release-assets.githubusercontent.com/signed"}, None)
                return io.BytesIO(b"data")
        with patch.object(archive.urllib.request, "build_opener", return_value=Opener()):
            with client.open_download({"id": 1}) as response:
                self.assertEqual(response.read(), b"data")
        self.assertIn("Authorization", calls[0].headers)
        self.assertNotIn("Authorization", calls[1].headers)
        with self.assertRaises(archive.ArchiveError):
            archive.safe_url("https://untrusted.example/signed", archive.DOWNLOAD_HOSTS)


if __name__ == "__main__":
    unittest.main()
