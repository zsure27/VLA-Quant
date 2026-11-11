import json
from pathlib import Path
import tempfile
import unittest

from qvla.groot_snapshot_20261111_v2 import SUITES,verify_snapshot,sha


class SnapshotContract(unittest.TestCase):
    def test_all_suite_revisions_are_valid_git_hashes(self):
        for _,_,revision,_ in SUITES.values():self.assertRegex(revision,r'^[0-9a-f]{40}$')
        self.assertEqual(SUITES['libero_object'][2],'f422266b016fd917127d3a0348062a5cd750df43')
        self.assertEqual(SUITES['libero_goal'][2],'17be45f5bf8a543594761b263c68cf33f33847bd')
        self.assertEqual(SUITES['libero_10'][2],'aa49078d5cc9ce72917bc4312f1ef12771f277de')

    def make_snapshot(self,folder):
        name,repo,revision,_=SUITES['libero_object'];root=Path(folder)/name;root.mkdir()
        weight=root/'model.safetensors';weight.write_bytes(b'synthetic identity test only')
        receipt=dict(repo=repo,revision=revision,files=1,bytes=weight.stat().st_size,
                     files_verified=[dict(path=weight.name,bytes=weight.stat().st_size,sha256=sha(weight))])
        (root/'DOWNLOAD_VERIFICATION.json').write_text(json.dumps(receipt))
        return root

    def test_correct_receipt_and_full_sha_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.make_snapshot(directory)
            self.assertEqual(verify_snapshot(root,'libero_object')['files'],1)
            original=(root/'model.safetensors').read_bytes()
            (root/'model.safetensors').write_bytes(b'X'+original[1:])
            with self.assertRaises(ValueError):verify_snapshot(root,'libero_object')

    def test_wrong_revision_and_duplicate_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.make_snapshot(directory);p=root/'DOWNLOAD_VERIFICATION.json';record=json.loads(p.read_text())
            record['revision']='0'*40;p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):verify_snapshot(root,'libero_object')
            record['revision']=SUITES['libero_object'][2];record['files']=2
            record['files_verified']*=2;p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):verify_snapshot(root,'libero_object')


if __name__=='__main__':unittest.main()
