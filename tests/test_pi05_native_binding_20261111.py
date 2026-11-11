import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qvla.pi05_native_binding_20261111_v1 import checkpoint_identity


class NativeIdentityTests(unittest.TestCase):
    def make_model(self, root):
        (root / 'params').mkdir()
        files = []
        for i in range(16):
            payload = bytes([i]) * (i + 1)
            path = root / 'params' / str(i)
            path.write_bytes(payload)
            files.append(dict(path='params/' + str(i), bytes=len(payload),
                              sha256=hashlib.sha256(payload).hexdigest()))
        record = dict(source='gs://openpi-assets/checkpoints/pi05_libero/',
                      objects=16, bytes=sum(x['bytes'] for x in files), files=files)
        (root / 'DOWNLOAD_VERIFICATION.json').write_text(json.dumps(record))
        return record

    def test_changed_content_rejected_even_with_same_size(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_model(root)
            valid = checkpoint_identity(root)
            self.assertEqual(len(valid['files']), 16)
            (root / 'params/0').write_bytes(b'X')
            with self.assertRaises(ValueError):
                checkpoint_identity(root)

    def test_manifest_cannot_escape_model_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = self.make_model(root)
            record['files'][0]['path'] = '../outside'
            (root / 'DOWNLOAD_VERIFICATION.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                checkpoint_identity(root)


if __name__ == '__main__':
    unittest.main()
