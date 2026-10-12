"""Dependency-light contracts for the real-environment CPU bridge."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bridge_contract", ROOT / "scripts/probe_qvla_baseline_protocol_cpu_20261012.py")
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


class BridgeContracts(unittest.TestCase):
    def test_archived_and_current_real_call_order(self):
        self.assertEqual(BRIDGE.sha(BRIDGE.HISTORICAL), BRIDGE.HISTORICAL_SHA)
        self.assertEqual(BRIDGE.inspect_seed_order(BRIDGE.HISTORICAL), ["env.seed", "seed_all"])
        self.assertEqual(BRIDGE.inspect_seed_order(ROOT / "overlays/openvla-oft/experiments/robot/libero/run_libero_eval.py"), ["seed_all", "env.seed"])

    def test_patch_restores_when_real_flow_fails(self):
        obj = SimpleNamespace(value="original")
        with self.assertRaises(RuntimeError):
            with BRIDGE.patched_attributes(obj, {"value": "patched"}):
                self.assertEqual(obj.value, "patched")
                raise RuntimeError("failure")
        self.assertEqual(obj.value, "original")

    def test_plan_is_bounded_and_does_not_admit_gpu(self):
        args = SimpleNamespace(tasks=10, resets=2, repeats=2, model_seed=7, env_seed=0,
                               historical_evaluator=BRIDGE.HISTORICAL, historical_sha=BRIDGE.HISTORICAL_SHA)
        value = BRIDGE.plan(args)
        self.assertEqual(value["episodes"], 120)
        self.assertEqual(value["reset_indices"], [0, 1])
        self.assertEqual(value["queries_per_episode"], 2)
        args.resets = 50
        with self.assertRaises(ValueError):
            BRIDGE.plan(args)

    def test_repeatability_and_cross_protocol_are_distinct(self):
        def ep(pixel):
            return {"initial_state_sha256": "same", "query_steps": {"query0": 10, "after_fixed_8": 18},
                    "queries": {"query0": {"model/pixels": {"sha256": pixel}}, "after_fixed_8": {"model/pixels": {"sha256": pixel}}}}
        runs = [{"protocol": p, "repeat": r, "episodes": {"0:0": ep(p)}}
                for p in BRIDGE.PROTOCOLS for r in range(2)]
        within, cross = BRIDGE.compare_runs(runs)
        self.assertTrue(all(v["repeat_equal"] for v in within.values()))
        self.assertTrue(all(v["changed_episodes"] == 1 for v in cross.values()))
        runs[-1]["episodes"]["0:0"] = ep("nonrepeatable")
        within, _ = BRIDGE.compare_runs(runs)
        self.assertFalse(within["corrected_paired"]["repeat_equal"])


if __name__ == "__main__":
    unittest.main()
