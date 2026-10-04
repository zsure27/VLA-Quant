"""Regression gates for wrong adapters, copied observations and cohort pooling."""
import hashlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from qvla.data_roles import check_disjoint, audit_training_inputs
from qvla.model_registry import validate_request, validate_adapter_shapes, validate_effective_plan
from qvla.paired_metrics import comparison
from qvla.state_metadata import MetadataUnpickler


class Contracts(unittest.TestCase):
    def test_copied_renamed_changed_action_is_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a.npz", Path(tmp)/"renamed.npz"
            fields = dict(image=np.zeros((2,2,3), np.uint8), wrist_image=np.ones((2,2,3), np.uint8),
                          state=np.arange(8, dtype=np.float32), instruction=np.array("test"))
            np.savez(a, **fields, action=np.zeros(7)); np.savez(b, **fields, action=np.ones(7))
            with self.assertRaises(ValueError): check_disjoint([a], [b])

    def test_spoofed_peft_train_role_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); sample=p/"sample-0.npz"
            np.savez(sample, image=np.zeros((1,1,3),np.uint8), wrist_image=np.zeros((1,1,3),np.uint8),
                     state=np.zeros(8,np.float32), instruction=np.array("test"))
            (p/"manifest.json").write_text(json.dumps({"samples":[{"file":sample.name,"role":"peft_train",
                "episode":1,"instruction":"test","sha256":hashlib.sha256(sample.read_bytes()).hexdigest()}]}))
            split=p/"split.json"; split.write_text(json.dumps({"episodes":[{"dataset_order_index":1,"role":"offline_final_holdout","instruction":"test"}]}))
            with self.assertRaises(ValueError): audit_training_inputs([sample],split)
            with self.assertRaises(ValueError): audit_training_inputs([sample],None)

    def test_different_frames_same_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i in range(2):
                parent=Path(tmp)/str(i); parent.mkdir(); sample=parent/"sample.npz"
                np.savez(sample,image=np.full((1,1,3),i,np.uint8),wrist_image=np.zeros((1,1,3),np.uint8),state=np.zeros(8),instruction=np.array("test"))
                (parent/"manifest.json").write_text(json.dumps({"samples":[{"file":sample.name,"dataset_order_index":5}]}))
                paths.append(sample)
            with self.assertRaises(ValueError): check_disjoint(paths[:1],paths[1:])

    def test_duplicate_cohort_rejected(self):
        row={"task_id":0,"init_state_index":5,"B":True,"A":False}
        with self.assertRaises(ValueError): comparison([row,row],"B","A")

    def test_dynamic_denominator_and_flips(self):
        rows=[{"task_id":t,"init_state_index":i,"B":True,"A":False} for t in range(2) for i in range(3)]
        r=comparison(rows,"B","A",200)
        self.assertEqual(r["task_cluster_bootstrap_95ci_fraction"],[1,1])
        self.assertEqual((r["rescue"],r["break"],r["net"]),(6,0,6))

    def test_nonboolean_label_rejected(self):
        with self.assertRaises(ValueError): comparison([{"task_id":0,"init_state_index":0,"B":1,"A":False}],"B","A")

    def test_planned_models_closed(self):
        with self.assertRaises(ValueError): validate_request(SimpleNamespace(model_id="B2"))

    def test_name_cannot_hide_adapter(self):
        args=SimpleNamespace(model_id="A3",method="awq",activation_bits=16,awq_scale_peft_state=None,awq_recovery_lora_state=Path("wrong"))
        with self.assertRaises(ValueError): validate_request(args)

    def test_wrong_layer_coverage_rejected(self):
        with self.assertRaises(ValueError): validate_adapter_shapes({"bad":(8,10)},"B0")

    def test_rank16_rejected(self):
        families=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj")
        shapes={f"language_model.model.layers.{l}.{f}.awq_recovery_lora.{j}.weight": ((8,16) if j==0 else (16,8)) for l in (18,19) for f in families for j in (0,1)}
        validate_adapter_shapes(shapes,"B0")
        shapes[next(iter(shapes))]=(16,16)
        with self.assertRaises(ValueError): validate_adapter_shapes(shapes,"B0")

    def test_bf16_must_have_zero_targets(self):
        with self.assertRaises(ValueError): validate_effective_plan("BF16",{"x":({},4,128)})

    def test_effective_groups_and_coverage(self):
        from qvla.model_registry import W4_LAYERS
        families=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj")
        targets={f"language_model.model.layers.{l}.{f}":({},4 if l in W4_LAYERS['A3'] else 2,128 if l in W4_LAYERS['A3'] else 64) for l in range(32) for f in families}
        for prefix,n,group in (("featurizer",93,64),("fused_featurizer",105,128)):
            targets.update({f"vision_backbone.{prefix}.layer{i}":({},2,group) for i in range(n)})
        validate_effective_plan("A3",targets)
        targets['vision_backbone.fused_featurizer.layer0']=({},2,64)
        with self.assertRaises(ValueError): validate_effective_plan("A3",targets)

    def test_metadata_pickle_blocks_code(self):
        # GLOBAL os.system must be rejected without calling it.
        with self.assertRaises(ValueError): MetadataUnpickler(io.BytesIO(b"cos\nsystem\n.")).load()


if __name__ == "__main__": unittest.main()
