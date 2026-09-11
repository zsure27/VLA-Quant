"""安装器回归：Linux/Windows 换行、重复安装、未知改动与写入前预检。"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_overlays", ROOT / "scripts/install_overlays.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src, self.dst, self.backup = self.root / "source", self.root / "destination", self.root / "backup"
        shutil.copytree(ROOT / "overlays/openvla-oft", self.src)
        for relative in installer.UPSTREAM_LF_SHA256:
            data = subprocess.check_output(["git", "show", "8d1e743:overlays/openvla-oft/" + relative], cwd=ROOT)
            target = self.dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def test_pinned_lf_and_crlf(self):
        for relative, expected in installer.UPSTREAM_LF_SHA256.items():
            target = self.dst / relative
            self.assertEqual(installer.normalized_hash(target), expected)
            target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(installer.normalized_hash(target), expected)
        installer.install(self.src, self.dst, self.backup)
        self.assertEqual(installer.plan_install(self.src, self.dst), [])

    def test_repeat_and_partial_install(self):
        relative = "experiments/robot/openvla_utils.py"
        shutil.copy2(self.src / relative, self.dst / relative)
        installer.install(self.src, self.dst, self.backup)
        installer.install(self.src, self.dst, self.backup)
        self.assertEqual(installer.plan_install(self.src, self.dst), [])

    def test_unknown_change_fails_before_any_write(self):
        relative = "experiments/robot/libero/run_libero_eval.py"
        target = self.dst / relative
        target.write_bytes(target.read_bytes() + b"\n# user change\n")
        before = {str(p.relative_to(self.dst)): p.read_bytes() for p in self.dst.rglob("*.py")}
        with self.assertRaisesRegex(ValueError, "未知源码修改"):
            installer.install(self.src, self.dst, self.backup)
        after = {str(p.relative_to(self.dst)): p.read_bytes() for p in self.dst.rglob("*.py")}
        self.assertEqual(before, after)
        self.assertFalse(self.backup.exists())


if __name__ == "__main__":
    unittest.main()
