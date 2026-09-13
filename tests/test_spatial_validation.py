"""用伪样本检查历史排除、轨迹隔离和不覆盖；不是实际数据验证。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("validation", Path(__file__).resolve().parents[1] / "scripts/prepare_spatial_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ValidationTest(unittest.TestCase):
    def test_exclusion_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, base, artifacts = root / "source", root / "artifacts/base", root / "artifacts"
            source.mkdir()
            (base / "samples").mkdir(parents=True)
            rows = []
            for i in range(128):
                name = f"sample-{i:04d}.npz"
                (source / name).write_bytes(str(i).encode())
                row = {"file": name, "suite": "libero_spatial_no_noops", "episode": i,
                       "sha256": validation.sha(source / name)}
                rows.append(row)
                if i < 64:
                    (base / "samples" / name).write_bytes(str(i).encode())
            (source / "manifest.json").write_text(json.dumps({"samples": rows}))
            (base / "samples/manifest.json").write_text(json.dumps({"samples": rows[:64]}))
            (artifacts / "old").mkdir()
            (artifacts / "old/manifest.json").write_text(json.dumps({"samples": {"renamed.npz": rows[72]["sha256"]}}))
            output = root / "output"
            result = validation.prepare(source, base, artifacts, output)
            self.assertEqual(len(result["samples"]), 32)
            self.assertEqual(result["samples"][0]["episode"], 73)
            self.assertEqual(result["samples"][0]["file"], "sample-1000.npz")
            self.assertTrue(all(r["episode"] >= 73 for r in result["samples"]))
            with self.assertRaises(FileExistsError):
                validation.prepare(source, base, artifacts, output)
            # 将所有源样本标记为已用后，必须拒绝选择而非降级复用。
            (artifacts / "old/manifest.json").write_text(json.dumps({"samples": {r["file"]: r["sha256"] for r in rows}}))
            with self.assertRaises(ValueError):
                validation.prepare(source, base, artifacts, root / "second")


if __name__ == "__main__":
    unittest.main()
