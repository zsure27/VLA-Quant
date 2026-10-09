"""Focused contract checks for the 300-episode B4 plan extension."""
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts import prepare_b4_pair_v3_plans as planner
from scripts import audit_b4_pair_v3 as auditor
from scripts.vla_stage_supervisor import validate_plan


def _stage(name, *command):
    return {"id": name, "cwd": str(Path.cwd()), "env": {}, "command": list(command)}


class PairV3PlanTests(TestCase):
 def test_final100_is_distinct_and_combined_only_after_it(self):
    self._check(Path.cwd() / ".local" / "test-b4-v3-plan")

 def _check(self, tmp_path):
    root = tmp_path / "session"
    output = tmp_path / "plans"
    def fake_v2(*_args):
        common = {"schema_version": "1.0", "control_root": str(output / "control"),
                  "terminal_state": "AWAITING_GATE_REVIEW", "question": "old", "next_plan": "old"}
        return {
            "first50": dict(common, plan_id="run-B4-first50-pair-v2", stages=[_stage("first50-B3", "python", "eval-first50")]),
            "next50": dict(common, plan_id="run-B4-next50-pair-v2", stages=[_stage("next50-B3", "python", "eval-next50")]),
            "last100": dict(common, plan_id="run-B4-last100-pair-v2", stages=[
                _stage("verify-next50", "python", "stage-records/verify-next50", "--require-prior", "next50"),
                _stage("last100-B3", "python", "eval-last100/B3", "--initial-state-offset", "30", "--num_trials_per_task", "10"),
                _stage("last100-B4", "python", "eval-last100/B4", "--initial-state-offset", "30", "--num_trials_per_task", "10"),
                _stage("audit-last100", "python", "scripts/audit_b4_pair_v2.py", "--phase", "last100", "last100-protocol-gate.json"),
                _stage("audit-combined", "python", "scripts/audit_b4_pair_v2.py", "--phase", "combined")])}
    with patch.object(planner, "build_v2", fake_v2):
        plans = planner.build_plans({}, root, output, "python", "expected")
    assert list(plans) == ["first50", "next50", "last100", "final100"]
    assert "audit-combined" not in [s["id"] for s in plans["last100"]["stages"]]
    final = plans["final100"]
    assert [s["id"] for s in final["stages"]] == ["verify-last100", "final100-B3", "final100-B4", "audit-final100", "audit-combined"]
    for case in ("B3", "B4"):
        command = next(s["command"] for s in final["stages"] if s["id"] == f"final100-{case}")
        assert command[command.index("--initial-state-offset") + 1] == "40"
        assert "eval-final100/" + case in command
    assert "audit_b4_pair_v3.py" in " ".join(final["stages"][-1]["command"])
    for plan in plans.values():
        validate_plan(plan)

 def test_combined_keeps_200_references_separate(self):
    parent = Path.cwd() / ".local"
    parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        for phase, start, stop in (("first50", 20, 25), ("next50", 25, 30),
                                   ("last100", 30, 40), ("final100", 40, 50)):
            rows = []
            for task in range(10):
                for reset in range(start, stop):
                    row = {"task_id": task, "init_state_index": reset, "B3": True, "B4": True}
                    if reset < 40:
                        row.update(A4=False, BF16=True, A0=True)
                    rows.append(row)
            (root / f"{phase}-protocol-gate.json").write_text(json.dumps({"paired_rows": rows}))
        with patch.object(auditor, "require_prior", return_value=None):
            result = auditor.combined(root)
        assert result["gate"] == "PASS_PROTOCOL_B4_PAIR_V3_COMBINED300"
        assert len(result["paired_rows"]) == 300
        assert result["archival_reference_20_39"]["episodes"] == 200
        assert all("BF16" not in row for row in result["paired_rows"] if row["init_state_index"] >= 40)
