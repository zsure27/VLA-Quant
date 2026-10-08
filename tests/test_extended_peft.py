"""CPU executable B2/B3 admission contracts; no claim about real GPU behavior."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from qvla.extended_peft import (target_names, descriptors, validate_shapes, load_state,
    zero_state, attach_state, trainable_contract, student_contract, sha)
from qvla.model_registry import W4_LAYERS, validate_request, validate_effective_plan
from scripts.train_extended_peft import copy_state, frozen_digest
from scripts.audit_extended_peft_eval import audit_case, audit, normalized_trace_state
from qvla.on_policy_capture import observation_hash


def toy():
    model = torch.nn.Module()
    targets = {}
    names = (Path(__file__).resolve().parent.parent / "configs/qvla-connected-422.txt").read_text().splitlines()
    for name in names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part): parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        linear = torch.nn.Linear(16, 16)
        parent.add_module(parts[-1], linear)
        targets[name] = ({"shape": [16, 16]}, 2, 128 if ".fused_featurizer." in name else 64)
    model.requires_grad_(False)
    return model, targets


def state_for(model, kind):
    modules = dict(model.named_modules())
    return {f"{n}.awq_recovery_lora.{j}.weight": v for n in target_names(kind)
            for j, v in enumerate(zero_state(modules[n], n))}


class ExtendedContracts(unittest.TestCase):
    def test_exact_target_names_and_protected_exclusion(self):
        self.assertEqual(len(target_names("B2")), 196)
        self.assertEqual(len(target_names("B3")), 224)
        self.assertTrue(all("patch_embed" not in n and "projector" not in n for n in target_names("B2")))

    def test_target_shape_bits_and_type(self):
        model, targets = toy(); modules = dict(model.named_modules())
        rows = descriptors("B2", modules, targets)
        self.assertEqual(sum(r["parameters"] for r in rows), 196 * 256)
        n = next(iter(target_names("B2")))
        targets[n] = ({"shape": [16, 16]}, 4, 64)
        with self.assertRaises(ValueError): descriptors("B2", modules, targets)
        targets[n] = ({"shape": [16, 16]}, 2, 64)
        modules[n] = torch.nn.Conv2d(16, 16, 1)
        with self.assertRaises(ValueError): descriptors("B2", modules, targets)

    def test_b2_zero_frozen_b1_gradients_and_reload(self):
        from qvla.model_registry import ADAPTER_LAYERS
        model, targets = toy(); modules = dict(model.named_modules())
        b1 = {f"language_model.model.layers.{l}.{f}.awq_recovery_lora.{j}.weight":
              (torch.randn(8, 16) if j == 0 else torch.randn(16, 8))
              for l in ADAPTER_LAYERS["B1"] for f in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj") for j in (0, 1)}
        attach_state(model, b1)
        state = state_for(model, "B2")
        x = torch.randn(3, 16)
        before = {n: modules[n](x).detach().clone() for n in target_names("B2")}
        attach_state(model, state, target_names("B2"))
        self.assertTrue(all(torch.equal(before[n], modules[n](x)) for n in target_names("B2")))
        named = trainable_contract(model, "B2")
        fingerprint = frozen_digest((("model", model),), set(named))
        opt = torch.optim.AdamW(list(named.values()), lr=1e-3)
        loss = sum(modules[n](x).square().mean() for n in target_names("B2"))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in named.values()))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n not in named))
        opt.step()
        self.assertEqual(fingerprint, frozen_digest((("model", model),), set(named)))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "visual.pt"; torch.save({n: v.detach().clone() for n, v in named.items()}, p)
            loaded = load_state(p, "B2", descriptors("B2", modules, targets))
            ref = {n: modules[n](x).detach().clone() for n in target_names("B2")}
            copy_state(model, {k: torch.zeros_like(v) for k, v in loaded.items()}); copy_state(model, loaded)
            self.assertTrue(all(torch.equal(ref[n], modules[n](x)) for n in target_names("B2")))

    def test_b3_all32_and_legacy_loader_stays_strict(self):
        from qvla.run_eval_official_quant import load_recovery_lora_state
        model, targets = toy(); state = state_for(model, "B3")
        self.assertEqual(len(state), 448)
        validate_effective_plan("B3", targets)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b3.pt"; torch.save(state, path)
            load_state(path, "B3", descriptors("B3", dict(model.named_modules()), targets))
            with self.assertRaises(RuntimeError): load_recovery_lora_state(path)
            bad = dict(state); bad.pop(next(iter(bad))); torch.save(bad, path)
            with self.assertRaises(ValueError): load_state(path, "B3")
            bad = dict(state); key = next(iter(bad)); bad[key] = torch.full_like(bad[key], float("nan")); torch.save(bad, path)
            with self.assertRaises(ValueError): load_state(path, "B3")

    def test_descriptor_and_rank_tamper(self):
        model, targets = toy(); state = state_for(model, "B2")
        shapes = {k: v.shape for k, v in state.items()}
        rows = descriptors("B2", dict(model.named_modules()), targets)
        bad = copy.deepcopy(rows); bad[0]["shape"] = [17, 16]
        with self.assertRaises(ValueError): validate_shapes(shapes, "B2", bad)
        shapes[next(iter(shapes))] = (16, 16)
        with self.assertRaises(ValueError): validate_shapes(shapes, "B2")

    def test_b3_wrong_backbone_and_b2_missing_visual_rejected(self):
        args = SimpleNamespace(model_id="B3", extended_peft_manifest=Path("registered"), method="awq",
            activation_bits=16, awq_scale_peft_state=None, awq_recovery_lora_state=Path("state"), awq_visual_lora_state=None,
            awq_scope="all", awq_candidate="w2-attention-primary-g64-stage-w4", weight_bits=2,
            awq_w4_layers="8,9,10,11,12,13,14,15,20,21,22,23")
        with self.assertRaises(ValueError): validate_request(args)
        args.awq_candidate="w2-attention-no-clip-primary-g64"; args.awq_w4_layers=""
        validate_request(args)
        args.model_id="B2"
        with self.assertRaises(ValueError): validate_request(args)

    def test_student_source_reset_query_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); rows=[]
            for t in range(10):
                for r in range(4):
                    for q in (0, 1):
                        path=directory/f"sample-{len(rows):05d}.npz"
                        np.savez(path, image=np.full((1,1,3), len(rows),np.uint8), wrist_image=np.zeros((1,1,3),np.uint8),
                                 state=np.zeros(8,np.float32),instruction=np.array(str(t)),action=np.zeros(7))
                        rows.append({"file":path.name,"task_id":t,"init_state_index":r,"query_in_episode":q,"sample_sha256":sha(path)})
            manifest={"role":"student_state_train","source_model_id":"A4","source_run_sha256":"registered",
                      "state_space":"policy_normalized_proprio", "samples":rows}
            p=directory/"manifest.json"; p.write_text(json.dumps(manifest)); student_contract(directory,"B3")
            manifest["source_model_id"]="A3"; p.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): student_contract(directory,"B3")
            manifest["source_model_id"]="A4"; rows[0]["init_state_index"]=20; p.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): student_contract(directory,"B3")

    def test_missing_trace_flag_rejects_before_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/"exit-code.txt").write_text("0")
            cmd=["python","evaluator","--model-id","B2","--initial-state-offset","20","--num_trials_per_task","1",
                 "--seed","0","--env-seed","1","--seed-protocol","paired","--task_suite_name","libero_spatial","--trace-actions"]
            (p/"invocation.json").write_text(json.dumps({"command":cmd}))
            with self.assertRaisesRegex(ValueError,"traces"): audit_case(p,"B2",20,1)

    def test_negative_result_passes_protocol_and_wrong_pair_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/"checkpoint"; checkpoint.mkdir()
            (checkpoint/"dataset_statistics.json").write_text(json.dumps({"libero_spatial_no_noops":{
                "proprio":{"q01":[-1.0]*8,"q99":[1.0]*8}}}))
            for case in ("B1","B2","A3","BF16","A0"):
                p=root/"eval-micro"/case; (p/"policy-observations").mkdir(parents=True)
                (p/"exit-code.txt").write_text("0")
                command=["python","evaluator","--model-id",case,"--initial-state-offset","20","--num_trials_per_task","1",
                         "--seed","0","--env-seed","1","--seed-protocol","paired","--task_suite_name","libero_spatial",
                         "--trace-actions","--trace-observations","--pretrained_checkpoint",str(checkpoint)]
                (p/"invocation.json").write_text(json.dumps({"command":command}))
                logs=[]; events=[]; queries=[]
                for t in range(10):
                    m={"task_id":t,"init_state_index":20,"init_state_sha256":str(t),"env_seed":1,"model_seed":t,"protocol":"paired"}
                    success=case!="B2"
                    logs += ["EPISODE_MANIFEST "+json.dumps(m), "Success: "+str(success)]
                    image=np.full((1,1,3),t,np.uint8); wrist=np.zeros((1,1,3),np.uint8); state=np.zeros(8,np.float32); action=np.zeros((8,7),np.float32)
                    source=p/"policy-observations"/f"query-{t:06d}.npz"
                    np.savez(source,image=image,wrist_image=wrist,state=state,instruction=np.array(str(t)),student_action=action)
                    events.extend([{"record_type":"episode_start","episode_serial":t},
                        {"record_type":"query","episode_serial":t,"query_in_episode":0,"chunk_execution_steps":8,
                         "executed_step_start":0,"task":str(t),"file":"policy-observations/"+source.name,"file_sha256":sha(source),
                         "observation_sha256":observation_hash(image,wrist,state,str(t))},
                        {"record_type":"episode_end","episode_serial":t,"queries":1,"success":success,"aborted":False}])
                    queries.append({"raw_policy_chunk":action.tolist(),"finite":True,"task":str(t),
                                    "state_space":"raw_proprio_before_get_action","state":state.tolist()})
                (p/"EVAL-fixture.txt").write_text("\n".join(logs))
                for filename,rows in (("on-policy-events.jsonl",events),("policy-queries.jsonl",queries)):
                    (p/filename).write_text("\n".join(json.dumps(r) for r in rows))
            result=audit(root,"B2","micro")
            self.assertEqual(result["gate"],"PASS_PROTOCOL_MICRO")
            self.assertEqual(result["primary"]["break"],10)
            p=root/"eval-micro/B2/EVAL-fixture.txt"
            p.write_text(p.read_text().replace('"init_state_sha256": "0"','"init_state_sha256": "changed"'))
            with self.assertRaisesRegex(ValueError,"pairing"): audit(root,"B2","micro")

    def test_proprio_space_is_explicit_and_normalized_only_once(self):
        from qvla.action_jacobian_batch import prepare_proprio
        stats={"q01":[-2.0]*8,"q99":[2.0]*8}
        raw=np.full(8,0.5,dtype=np.float32)
        policy=normalized_trace_state(raw,stats)
        self.assertTrue(np.array_equal(prepare_proprio(policy,stats,"policy_normalized_proprio"),policy))
        self.assertTrue(np.array_equal(prepare_proprio(raw,stats,"raw_proprio_before_get_action").astype(np.float32),policy))
        self.assertFalse(np.array_equal(prepare_proprio(policy,stats,"raw_proprio_before_get_action").astype(np.float32),policy))
        with self.assertRaisesRegex(ValueError,"Unknown proprio state space"):
            prepare_proprio(raw,stats,"unspecified")
        with self.assertRaisesRegex(ValueError,"malformed"):
            prepare_proprio(np.full(8,2.0,dtype=np.float32),stats,"policy_normalized_proprio")

    def test_plans_are_bounded_trace_complete_and_do_not_train_visual_b3(self):
        from scripts.prepare_extended_peft_plans import build_plan
        from scripts.vla_stage_supervisor import validate_plan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/"checkpoint"; checkpoint.mkdir(); (checkpoint/"config.json").write_text("{}")
            data=root/"data"; data.mkdir(); (data/"manifest.json").write_text("{}")
            blob=root/"blob"; blob.write_text("fixture")
            m={k:str(blob) for k in ("w2_g128","w2_g64","w4","b1_adapter","trajectory_split")}
            m.update(checkpoint=str(checkpoint),student80=str(data),calibration80=str(data),oft_root="/oft",libero_root="/libero",official_root="/official")
            for kind in ("B2","B3"):
                out=root/kind; out.mkdir()
                plan=build_plan(kind,m,root/"20261008-test",out,"python","hostname")
                validate_plan(plan)
                self.assertEqual(plan["terminal_state"],"AWAITING_GATE_REVIEW")
                self.assertEqual(plan["remaining_long_evaluation"],"LOCKED")
                for s in plan["stages"]:
                    cmd=s["command"]; child=cmd[cmd.index("--")+1:]
                    if "--local_log_dir" in child:
                        self.assertIn("--trace-observations",child); self.assertIn("--trace-actions",child)
                        self.assertLessEqual(int(child[child.index("--num_trials_per_task")+1]),5)
                ids=[s["id"] for s in plan["stages"]]
                self.assertLess(ids.index("smoke10"),ids.index("train1000"))
                self.assertLess(ids.index("audit-micro"),ids.index("first50-"+kind))
                if kind=="B3": self.assertLess(ids.index("prepare-A4-student80"),ids.index("smoke10"))

    def test_dynamic_artifact_smoke_and_training_overlap_rejected(self):
        from qvla.extended_peft import ARTIFACT_SOURCES, VERSION, canonical_sha, validate_artifact, ROOT
        model, targets=toy(); rows=descriptors("B3",dict(model.named_modules()),targets); state=state_for(model,"B3")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); adapter=root/"adapter.pt"; torch.save(state,adapter)
            training=root/"manifest.json"; training.write_text(json.dumps({"samples":[{"task_id":0,"init_state_index":0}]}))
            record={"version":VERSION,"model_id":"B3","steps":1000,"gate":"PASS_TRAIN_CONTRACT",
                    "spec_sha256":sha(ROOT/"configs/experiments/b2_b3_v2_proprio_20261008.json"),
                    "source_sha256":{p:sha(ROOT/p) for p in ARTIFACT_SOURCES},"zero_output_equal":True,"reload_output_equal":True,
                    "smoke_artifact_sha256":"smoke","frozen_before_sha256":"same","frozen_after_sha256":"same",
                    "targets":rows,"target_sha256":canonical_sha(rows),"training_manifest_sha256":sha(training),
                    "training_state_space":"policy_normalized_proprio",
                    "adapter_sha256":sha(adapter),"tensor_shapes":{k:list(v.shape) for k,v in state.items()}}
            manifest=root/"artifact.json"; manifest.write_text(json.dumps(record))
            args=SimpleNamespace(model_id="B3",extended_peft_manifest=manifest,recovery_training_manifest=training,
                                 awq_recovery_lora_state=adapter,task_suite_name="libero_spatial",initial_state_offset=20,num_trials_per_task=1)
            validate_artifact(args)
            args.initial_state_offset=0
            with self.assertRaisesRegex(ValueError,"overlap"): validate_artifact(args)
            args.initial_state_offset=20; record["steps"]=10; manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"full-training"): validate_artifact(args)


if __name__ == "__main__": unittest.main()
