"""使用临时伪样本验证选择与不覆盖保护，不模拟模型或GPU结果。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SplitTest(unittest.TestCase):
    def test_split_and_no_overwrite(self):
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_spatial_audit.py"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            rows = []
            for i in range(70):
                name = f"sample-{i:04d}.npz"
                (source / name).write_bytes(f"fixture-{i}".encode())
                rows.append({"file": name, "suite": "libero_spatial_no_noops", "episode": i})
            (source / "manifest.json").write_text(json.dumps({"samples": rows}))
            cmd = [sys.executable, str(script), "--source", str(source), "--output", str(root / "output")]
            result = subprocess.run(cmd, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            selected = json.loads((root / "output/manifest.json").read_text(encoding="utf-8"))["samples"]
            self.assertEqual(len(selected), 64)
            self.assertEqual(sum(r["role"] == "calibration" for r in selected), 32)
            self.assertEqual(len({r["episode"] for r in selected}), 64)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
