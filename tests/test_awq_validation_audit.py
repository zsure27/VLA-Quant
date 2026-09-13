"""验证冻结输入检查能拒绝样本污染和多余文件。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_group_validation as audit


class ValidationAuditTest(unittest.TestCase):
    def test_frozen_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,group,val=[Path(tmp)/n for n in ('base','group','validation')]
            for p in (base/'samples',val/'samples',val/'teacher'):
                p.mkdir(parents=True)
            rows=[]
            for i in range(32):
                name=f'sample-{1000+i}.npz'
                (val/'samples'/name).write_bytes(str(i).encode())
                rows.append(dict(file=name,suite='libero_spatial_no_noops',episode=i,sha256=str(i)))
            original=[dict(file='old.npz',suite='libero_spatial_no_noops',episode=100,sha256='old')]
            (base/'samples/manifest.json').write_text(json.dumps(dict(samples=original)))
            (val/'samples/manifest.json').write_text(json.dumps(dict(samples=rows)))
            (val/'teacher/manifest.json').write_text(json.dumps(dict(samples={r['file']:r['sha256'] for r in rows})))
            def fake_sha(p):
                if p.name=='w2.pt':
                    return '4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc'
                if p.name=='w2-g64.pt':
                    return '947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8'
                return p.read_text()
            with patch.object(audit,'sha',side_effect=fake_sha):
                self.assertEqual(audit.preflight(base,group,val)['status'],'PASS')
                (val/'samples/sample-extra.npz').write_bytes(b'extra')
                with self.assertRaises(ValueError):
                    audit.preflight(base,group,val)

