"""B4 causal/data/source admission and CPU joint gradient/reload checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from qvla.b4_joint_peft import (ROOT, SPEC, SOURCES, target_names, descriptors, validate_shapes,
    load_state, trainable_contract, attach_state, zero_state, validate_request, closure_contract, sha)
from qvla.extended_peft import target_names as old_targets
from scripts.train_b4_joint_peft import copy_state, frozen_digest
from tests.test_extended_peft import toy


class JointContracts(unittest.TestCase):
    def test_exact_union_parameters_and_archived_source_bytes(self):
        self.assertEqual(len(target_names()), 420)
        self.assertFalse(old_targets("B2") & old_targets("B3"))
        directory = ROOT / "results/experiments/p2-shared-peft/20261008-046-b2-b3"
        b3 = json.loads((directory / "B3-train1000-artifact.json").read_text())
        b2 = json.loads((directory / "B2-train1000-artifact.json").read_text())
        rows = b3["targets"] + b2["targets"]
        shapes = dict(b3["tensor_shapes"], **b2["tensor_shapes"])
        validate_shapes(shapes, rows=rows)
        self.assertEqual(sum(r["parameters"] for r in rows), 26710528)
        for name, expected in b3["source_sha256"].items():
            self.assertEqual(sha(ROOT / name), expected, name)
        spec = json.loads(SPEC.read_text())
        self.assertEqual(sha(directory / "B3-train1000-artifact.json"), spec["B3_training_artifact_sha256"])

    def test_joint_zero_gradient_frozen_and_reload(self):
        model, targets = toy()
        modules = dict(model.named_modules())
        x = torch.randn(3, 16)
        base = {n: modules[n](x).detach().clone() for n in target_names()}
        language = {f"{n}.awq_recovery_lora.{j}.weight":
            (torch.randn(8, 16) * .01 if j == 0 else torch.randn(16, 8) * .01)
            for n in old_targets("B3") for j in (0, 1)}
        attach_state(model, language)
        language_outputs = {n: modules[n](x).detach().clone() for n in target_names()}
        visual = {f"{n}.awq_recovery_lora.{j}.weight": v for n in old_targets("B2")
                  for j, v in enumerate(zero_state(modules[n], n))}
        attach_state(model, visual, old_targets("B2"))
        for n in old_targets("B3"): modules[n].awq_recovery_lora.requires_grad_(True)
        self.assertTrue(all(torch.equal(language_outputs[n], modules[n](x)) for n in target_names()))
        initial = dict(language, **visual)
        copy_state(model, {k: torch.zeros_like(v) for k, v in initial.items()})
        self.assertTrue(all(torch.equal(base[n], modules[n](x)) for n in target_names()))
        copy_state(model, initial)
        named = trainable_contract(model)
        self.assertEqual(len(named), 840)
        before = frozen_digest((("model", model),), set(named))
        opt = torch.optim.AdamW(list(named.values()), lr=1e-4)
        sum(modules[n](x).square().mean() for n in target_names()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in named.values()))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n not in named))
        opt.step()
        self.assertEqual(before, frozen_digest((("model", model),), set(named)))
        state = {k: v.detach().clone() for k, v in named.items()}
        expected = {n: modules[n](x).detach().clone() for n in target_names()}
        (ROOT / "backups").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "backups") as directory:
            path = Path(directory) / "joint.pt"
            torch.save(state, path)
            restored = load_state(path, rows=descriptors("B4", modules, targets))
            copy_state(model, {k: torch.zeros_like(v) for k, v in state.items()})
            copy_state(model, restored)
            self.assertTrue(all(torch.equal(expected[n], modules[n](x)) for n in target_names()))

    def test_missing_vision_wrong_rank_and_recipe_rejected(self):
        shapes = {f"{n}.awq_recovery_lora.{j}.weight": [8, 16] if j == 0 else [16, 8]
                  for n in target_names() for j in (0, 1)}
        validate_shapes(shapes)
        bad = copy.deepcopy(shapes); bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError): validate_shapes(bad)
        bad = copy.deepcopy(shapes); bad[next(iter(bad))] = [16, 16]
        with self.assertRaises(ValueError): validate_shapes(bad)
        args = SimpleNamespace(model_id="B4", method="awq", activation_bits=16, awq_scale_peft_state=None,
            awq_recovery_lora_state=Path("joint.pt"), awq_visual_lora_state=None, extended_peft_manifest=Path("artifact.json"),
            awq_scope="all", weight_bits=2, awq_candidate="w2-attention-no-clip-primary-g64", awq_w4_layers="")
        validate_request(args)
        args.awq_w4_layers="18,19"
        with self.assertRaises(ValueError): validate_request(args)

    def test_incomplete_prior_closure_cannot_admit_training(self):
        (ROOT / "backups").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "backups") as directory:
            receipt=Path(directory)/"receipt.json"
            receipt.write_text(json.dumps({"gate":"INCOMPLETE_LOCAL_ARCHIVE"}))
            with self.assertRaisesRegex(ValueError,"prior closure"):
                closure_contract({"prior_closure_receipt":str(receipt)})

    def test_eval_fork_preserves_all_rollout_mechanics(self):
        # Audit the whole fork: normalization, reset order and chunk execution
        # must match the archived evaluator; only B4 admission/loading differs.
        old=(ROOT/"qvla/run_eval_official_quant.py").read_text(encoding="utf-8")
        new=(ROOT/"qvla/run_eval_b4_joint.py").read_text(encoding="utf-8").split("\n",1)[1]
        expected=old.replace('from qvla.model_registry import MODEL_IDS, validate_request',
            'from qvla.b4_joint_peft import MODEL_IDS, validate_request')
        expected=expected.replace('from qvla.model_registry import validate_profile_provenance',
            'from qvla.b4_joint_peft import validate_profile_provenance')
        expected=expected.replace('from qvla.model_registry import validate_effective_plan',
            'from qvla.b4_joint_peft import validate_effective_plan')
        expected=expected.replace('from qvla.extended_peft import ', 'from qvla.b4_joint_peft import ').replace('"B3"','"B4"')
        self.assertEqual(new,expected)

    def test_plan_count_traces_and_hard_gate_boundaries(self):
        from scripts.prepare_b4_joint_plans import build_plans
        from scripts.vla_stage_supervisor import validate_plan
        (ROOT / "backups").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "backups") as directory:
            d=Path(directory); plans_dir=d/"plans"; plans_dir.mkdir()
            materials={k:str(d/k) for k in ("checkpoint", "trajectory_split", "w2_g128", "w2_g64", "w4",
                "prior_closure_receipt", "b3_train_dir", "student80", "calibration80", "oft_root", "libero_root", "official_root")}
            with patch("scripts.prepare_b4_joint_plans.sha", return_value="fixture"):
                plans=build_plans(materials, d/"session", plans_dir, "python", "verified-host")
            episodes={c:0 for c in ("B3","B4","A4","BF16","A0")}
            locks=set()
            for phase, plan in plans.items():
                validate_plan(plan)
                self.assertEqual(plan["terminal_state"],"AWAITING_GATE_REVIEW")
                for stage in plan["stages"]:
                    cmd=stage["command"]; locks.add(cmd[cmd.index("--lock")+1]); argv=cmd[cmd.index("--")+1:]
                    if "--local_log_dir" in argv:
                        self.assertIn("--trace-observations",argv); self.assertIn("--trace-actions",argv)
                        if not stage["id"].startswith("micro-"):
                            episodes[argv[argv.index("--model-id")+1]]+=10*int(argv[argv.index("--num_trials_per_task")+1])
                ids=[s["id"] for s in plan["stages"]]
                if phase=="first50":
                    self.assertLess(ids.index("smoke10"),ids.index("train1000"))
                    self.assertLess(ids.index("audit-micro"),ids.index("first50-B4"))
                else: self.assertTrue(ids[0].startswith("verify-"))
            self.assertEqual(set(episodes.values()),{200})
            self.assertEqual(len(locks),1)

    def test_negative_success_and_pairing_mismatch_gate(self):
        from scripts.audit_b4_joint_eval import audit, CASES
        keys=[(t,20,str(t),1,t,"paired") for t in range(10)]
        def result(path, model_id, offset, count):
            return {"keys":keys[:],"first":{i:str(i) for i in range(10)},
                "success":[model_id!="B4"]*10,"query_count":10,"sha256":{"fixture":"sha"}}
        with patch("scripts.audit_b4_joint_eval.audit_case",side_effect=result):
            gate=audit(Path("fixture"),"micro")
        self.assertEqual(gate["gate"],"PASS_PROTOCOL_B4_MICRO")
        self.assertEqual(gate["primary"]["break"],10)
        def mismatch(path, model_id, offset, count):
            r=result(path,model_id,offset,count)
            if model_id=="B4": r["first"][0]="different"
            return r
        with patch("scripts.audit_b4_joint_eval.audit_case",side_effect=mismatch):
            with self.assertRaisesRegex(ValueError,"manifest keys or first"):
                audit(Path("fixture"),"micro")


if __name__ == "__main__": unittest.main()
