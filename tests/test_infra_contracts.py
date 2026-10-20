"""CPU tests for risks introduced by the new orchestration layer; no model or server."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qvla.infra.common import canonical, read_json, write_json
from qvla.infra.config import validate
from qvla.infra.lifecycle import quota_decision
from qvla.infra.metrics import summarize, paired
from qvla.infra.plan import compile_plan, expand
from qvla.infra.report import record_round

ROOT = Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / "configs/infra/awq_multisuite.example.json")

    def test_multisuite_commands_cannot_silently_use_spatial_or_change_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = compile_plan(self.config, ROOT, Path(temp) / "run")
        self.assertEqual(len(plan["stages"]), 4)
        for suite, stage in zip(self.config["evaluation"]["suites"], plan["stages"]):
            args = stage["argv"]
            self.assertEqual(args[args.index("--task_suite_name") + 1], suite["name"])
            self.assertEqual(args[args.index("--num_trials_per_task") + 1], "1")
        self.config["quantization"]["recipe_argv"] = ["--seed", "2"]
        with self.assertRaises(ValueError):
            compile_plan(self.config, ROOT, ROOT / ".local/unexecuted")

    def test_heldout_overlap_and_unsupported_binding_rejected(self):
        self.config["evaluation"]["suites"][0]["role"] = "offline_final_holdout"
        with self.assertRaises(ValueError):
            validate(self.config)
        self.setUp()
        self.config["data"]["training"] = [{"suite": "libero_spatial", "role": "training",
            "namespace": "simulator_reset", "task_ids": [0], "resets": [5], "manifest": "missing.json", "sha256": "0" * 64}]
        with self.assertRaises(ValueError):
            validate(self.config)
        self.setUp()
        self.config["finetuning"]["method"] = "routing"
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_quota_boundaries_freshness_and_no_clock_inferred_refresh(self):
        control = self.config["control"]
        current = datetime.now(timezone.utc)
        observation = {"source": "codex_api", "window_minutes": 300, "observed_at": current.isoformat(),
                       "remaining_percent": 10, "resets_at": (current + timedelta(hours=1)).timestamp()}
        self.assertEqual(quota_decision(observation, control, current), "CLOSE")
        observation["remaining_percent"] = 15
        self.assertEqual(quota_decision(observation, control, current), "STOP_EXPANSION")
        observation["remaining_percent"] = 16
        self.assertEqual(quota_decision(observation, control, current), "CONTINUE")
        observation["observed_at"] = (current - timedelta(minutes=20)).isoformat()
        with self.assertRaises(ValueError):
            quota_decision(observation, control, current)
        observation["observed_at"] = current.isoformat()
        observation["resets_at"] = current.timestamp() - 1
        with self.assertRaises(ValueError):
            quota_decision(observation, control, current)

    def test_cohort_identity_interruption_and_seed_mismatch(self):
        rows = read_json(ROOT / "configs/infra/episodes.example.json")
        result = summarize(rows)
        self.assertEqual(len(result["scores"]), 2)
        self.assertEqual(paired(rows, "reference", "candidate")[0]["break"], 1)
        with self.assertRaises(ValueError):
            summarize(rows + [dict(rows[0], round_id="another-round")])
        changed = copy.deepcopy(rows)
        changed[1]["env_seed"] = 9
        with self.assertRaises(ValueError):
            paired(changed, "reference", "candidate")
        rows[1].update(status="interrupted", success=None)
        result = summarize(rows)
        self.assertEqual(len(result["excluded"]), 1)
        self.assertEqual(len(result["scores"]), 1)

    def test_rounds_render_into_one_task_result_before_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            record_round(repo, self.config, "round1", {"state": "INTERRUPTED"})
            record_round(repo, self.config, "round2", {"state": "EXECUTION_COMPLETE"})
            docs = list((repo / "Ex_Reports").glob("*.md"))
            self.assertEqual(len(docs), 1)
            text = docs[0].read_text(encoding="utf-8")
            self.assertLess(text.index("整合后的实验结果与分析"), text.index("多轮运行日志"))
            self.assertIn("round1", text)
            self.assertIn("round2", text)

    def test_backup_or_reporting_failure_still_attempts_shutdown(self):
        # Simulate Linux/control failures entirely in memory: never signal a real process.
        import qvla.infra.runner as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config["control"]["persistent_root"] = str(root)
            plan = compile_plan(self.config, ROOT, root / "run")
            auth = {"authorized": True, "shutdown_authorized": True, "hostname": "test-host",
                    "server_id": "091", "config_sha256": canonical(self.config), "round_id": "r1",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
            write_json(root / "auth.json", auth)
            fake_os = SimpleNamespace(name="posix", open=os.open, close=os.close,
                                      O_CREAT=os.O_CREAT, O_EXCL=os.O_EXCL, O_WRONLY=os.O_WRONLY)
            with patch.object(runner, "os", fake_os), patch.object(runner.socket, "gethostname", return_value="test-host"), \
                 patch.object(runner, "verify_plan", side_effect=ValueError("missing model")), \
                 patch.object(runner, "run_command", side_effect=ValueError("upload auth unavailable")), \
                 patch.object(runner, "record_round", side_effect=ValueError("report unavailable")), \
                 patch.object(runner, "native_shutdown") as shutdown:
                result = runner.execute(plan, root / "auth.json", "r1", root / "quota.json")
            shutdown.assert_called_once()
            self.assertTrue(result["closure_errors"])

    def test_completed_command_plan_preserves_argv_and_closes_once(self):
        import qvla.infra.runner as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.config["control"]["persistent_root"] = str(root)
            plan = compile_plan(self.config, ROOT, root / "run")
            auth = {"authorized": True, "shutdown_authorized": True, "hostname": "test-host",
                    "server_id": "091", "config_sha256": canonical(self.config), "round_id": "r2",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
            write_json(root / "auth.json", auth)
            write_json(root / "quota.json", {"source": "codex_api", "window_minutes": 300,
                "observed_at": datetime.now(timezone.utc).isoformat(), "remaining_percent": 90,
                "resets_at": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()})
            fake_os = SimpleNamespace(name="posix", open=os.open, close=os.close, environ=os.environ, pathsep=os.pathsep,
                                      O_CREAT=os.O_CREAT, O_EXCL=os.O_EXCL, O_WRONLY=os.O_WRONLY)
            process = SimpleNamespace(poll=lambda: 0, returncode=0, pid=123456)
            with patch.object(runner, "os", fake_os), patch.object(runner.socket, "gethostname", return_value="test-host"), \
                 patch.object(runner, "verify_plan"), patch.object(runner, "observe", return_value={"disk_free_bytes": 32*1024**3}), \
                 patch.object(runner.subprocess, "check_output", return_value=""), \
                 patch.object(runner.subprocess, "Popen", return_value=process) as spawn, \
                 patch.object(runner, "run_command") as close_command, patch.object(runner, "record_round"), \
                 patch.object(runner, "native_shutdown") as shutdown:
                result = runner.execute(plan, root / "auth.json", "r2", root / "quota.json")
            self.assertEqual(spawn.call_count, 4)
            self.assertTrue(all(r["exit_code"] == 0 for r in result["stages"]))
            self.assertEqual(result["stages"][0]["argv"], plan["stages"][0]["argv"])
            self.assertEqual(close_command.call_count, 2)
            shutdown.assert_called_once()
            self.assertFalse(result["closure_errors"])


if __name__ == "__main__":
    unittest.main()
