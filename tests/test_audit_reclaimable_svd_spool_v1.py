"""CPU contracts for dangerous path and independently verified backup admission."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('spool_reclaim', Path(__file__).resolve().parents[1]
                                            / 'scripts/audit_reclaimable_svd_spool_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReclaimAdmission(unittest.TestCase):
    def test_path_and_relative_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                m.require_exact_directory(root, root / 'expected')
        for name in ('../outside', '/absolute', 'x/../y', r'x\outside', 'x:y'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.safe_relative(name)

    def test_receipt_must_bind_every_file_and_manifest(self):
        manifest = {'schema': 'vla-svd-spool-content-v1', 'target': str(m.TARGET),
                    'files': [{'relative_path': 'layer/000.pt', 'bytes': 3, 'sha256': 'a' * 64}]}
        valid = {'gate': 'PASS_LOCAL_SVD_SPOOL_BACKUP_ALL_FILES', 'all_files_verified': True,
                 'independent_local_copy': True, 'server_hostname': 'clone091',
                 'target': str(m.TARGET), 'content_manifest_sha256': m.canonical_sha(manifest),
                 'files': manifest['files'], 'local_archive_path': 'D:/local/spool.tar',
                 'archive_sha256': 'b' * 64, 'archive_bytes': 10240}
        m.validate_backup_receipt(valid, manifest, 'clone091')
        for change in ({'all_files_verified': False}, {'independent_local_copy': False},
                       {'files': []}, {'content_manifest_sha256': 'c' * 64},
                       {'server_hostname': 'wrong'}, {'local_archive_path': '/root/archive.tar'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_backup_receipt({**valid, **change}, manifest, 'clone091')


if __name__ == '__main__':
    unittest.main()
