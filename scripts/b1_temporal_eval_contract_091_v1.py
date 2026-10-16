"""Independent trained temporal80 artifact and actual-policy admission; old loaders unchanged."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(os.environ["VLA_FROZEN_SOURCE"])
sys.path.insert(0,str(ROOT))
VERSION = "b1-temporal80-eval-v1-20261017"
TRAIN_VERSION = "b1-temporal80-v1-20261017"
MODELS = {"early80":"B1-temporal-Early80-v1", "phase80":"B1-temporal-Phase80-v1"}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def validate_metadata(record, variant, training, spec_sha):
    from qvla.extended_peft import validate_shapes, target_names, canonical_sha
    if (record.get("version") != TRAIN_VERSION or record.get("model_id") != "B3"
            or record.get("temporal_variant") != variant or variant not in MODELS
            or record.get("steps") != 1000 or record.get("gate") != "PASS_TRAIN_CONTRACT"
            or record.get("initialization") != "response-svd" or record.get("spec_sha256") != spec_sha
            or record.get("parameters") != 19988480 or record.get("training_state_space") != "policy_normalized_proprio"
            or not record.get("smoke_artifact_sha256") or not record.get("zero_output_equal")
            or not record.get("reload_output_equal") or not record.get("frozen_before_sha256")
            or record["frozen_before_sha256"] != record.get("frozen_after_sha256")):
        raise ValueError("Wrong temporal trained identity/budget/frozen/zero/reload contract")
    rows=record['targets']
    if len(rows)!=224 or {r['name'] for r in rows}!=target_names('B3') or canonical_sha(rows)!=record['target_sha256']:
        raise ValueError('Wrong224 language target descriptors')
    validate_shapes(record['tensor_shapes'],'B3',rows)
    samples=training['samples']
    if (training.get('variant')!=variant or training.get('role')!='student_state_train'
            or training.get('source_model_id')!='A4' or training.get('state_space')!='policy_normalized_proprio'
            or training.get('success_filtering') is not False or training.get('holdout_touched') is not False
            or len(samples)!=80 or {(r['task_id'],r['init_state_index'],r['slot_in_episode']) for r in samples}
               !={(t,r,s) for t in range(10) for r in range(4) for s in (0,1)}):
        raise ValueError('Temporal training role/coverage changed')

def validate_provenance(args):
    from qvla.extended_peft import registry,load_state
    if sha(args.trained_artifact)!=args.trained_artifact_sha256:
        raise ValueError('Explicit trusted trained artifact SHA mismatch')
    record=json.loads(args.trained_artifact.read_text())
    training=json.loads(args.recovery_training_manifest.read_text())
    validate_metadata(record,args.candidate,training,sha(args.training_spec))
    if sha(args.recovery_training_manifest)!=record['training_manifest_sha256'] or sha(args.adapter_state)!=record['adapter_sha256']:
        raise ValueError('Training or trained adapter bytes changed')
    for name,expected in record['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Training source changed: '+name)
    smoke=args.trained_artifact.parent.parent/'steps-10/artifact.json'
    if sha(smoke)!=record['smoke_artifact_sha256']:raise ValueError('Smoke ancestry changed')
    for name,expected in record['reuse_files'].items():
        if sha(args.trained_artifact.parent/name)!=expected:raise ValueError('Saved training materials changed')
    pair=json.loads(args.training_pair_gate.read_text())
    if pair['gate']!='PASS_B1_TEMPORAL_TRAIN_PAIR' or pair['artifacts'][args.candidate]!=args.trained_artifact_sha256:
        raise ValueError('Missing completed paired training gate')
    load_state(args.adapter_state,'B3',record['targets'])
    hashes=registry()['profile_sha256']
    if len(args.profile)!=1 or sha(args.profile[0])!=hashes['w2_g128'] or sha(args.awq_primary_group64_profile)!=hashes['w2_g64']:
        raise ValueError('A4 profile changed')
    if (args.model_id!=MODELS[args.candidate] or (args.method,args.weight_bits,args.activation_bits,args.awq_candidate)
           !=('awq',2,16,'w2-attention-no-clip-primary-g64')
           or (args.task_suite_name,args.seed,args.env_seed,args.seed_protocol)!=('libero_spatial',0,1,'paired')
           or args.initial_state_offset<5 or args.num_trials_per_task<1 or args.initial_state_offset+args.num_trials_per_task>50
           or not args.trace_actions or not args.trace_observations):
        raise ValueError('Unregistered temporal evaluation condition')
    if sha(args.ptq_reference_sample)!=args.ptq_reference_sha256:
        raise ValueError('Canonical matching PTQ observation changed')
    return record

def matching_ptq_bridge(gate, observation, task, model, cfg, args):
    import numpy as np
    from qvla.extended_proprio import prepare_proprio
    with np.load(args.ptq_reference_sample,allow_pickle=False) as z:
        normalized=prepare_proprio(np.asarray(observation['state']).copy(),model.norm_stats[cfg.unnorm_key]['proprio'],'raw_proprio_before_get_action')
        if (str(z['instruction'].item())!=task or not np.array_equal(z['image'],np.asarray(observation['full_image']))
                or not np.array_equal(z['wrist_image'],np.asarray(observation['wrist_image']))
                or not np.array_equal(z['state'],normalized)
                or not np.array_equal(z['student_action'],np.asarray(gate['zero_branch_action'],dtype=np.float32))):
            raise ValueError('Real zero branch/input not exactly matching canonical PTQ at same task/reset')
    gate['matching_PTQ_input_and_zero_action_equal']=True
    gate['ptq_reference_sha256']=args.ptq_reference_sha256

def frozen_digest(modules):
    """Match the archived model/head/proprio stream, excluding only adapter keys."""
    import torch
    digest = hashlib.sha256()
    for prefix, model in modules:
        for name, value in sorted(model.state_dict().items()):
            if prefix == "model" and ".awq_recovery_lora." in name:
                continue
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite frozen tensor: {prefix}.{name}")
            digest.update(f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def copy_factors(model, state, zero=False):
    import torch
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise ValueError("Runtime adapter shape changed")
            parameter.copy_(torch.zeros_like(parameter) if zero and ".awq_recovery_lora.1." in name
                            else value.to(parameter))


def first_query_contract(model, cfg, observation, task, kwargs, get_action, record,
                         state, state_path):
    """No training; raw observations go through deployed get_action exactly once.

    Probe calls use independent deep copies and restore every RNG stream, so
    they cannot double-normalize the actual rollout observation or reseed it.
    """
    import copy
    import random
    import numpy as np
    import torch
    from qvla.extended_peft import load_state
    from qvla.extended_proprio import prepare_proprio
    raw = copy.deepcopy(observation)
    modules = (("model", model), ("action_head", kwargs["action_head"]),
               ("proprio", kwargs["proprio_projector"]))
    for _name, module in modules:
        module.requires_grad_(False)
        module.eval()
    before = frozen_digest(modules)
    if before != record["frozen_before_sha256"]:
        raise ValueError("Runtime A4/head/proprio bytes differ from the archived frozen base")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cpu_rng = torch.get_rng_state()
    try:
        with torch.no_grad():
            reference_observation = copy.deepcopy(raw)
            reference = np.asarray(get_action(cfg, model, reference_observation, task, **kwargs)).copy()
            normalized = prepare_proprio(np.asarray(raw["state"]).copy(),
                model.norm_stats[cfg.unnorm_key]["proprio"], "raw_proprio_before_get_action")
            if not np.array_equal(np.asarray(reference_observation["state"]), normalized):
                raise ValueError("Deployed policy proprio does not match exactly one normalization")
            if state is not None:
                copy_factors(model, state, zero=True)
                zero = np.asarray(get_action(cfg, model, copy.deepcopy(raw), task, **kwargs)).copy()
                reloaded = load_state(state_path, "B3", record["targets"])
                copy_factors(model, reloaded)
            else:
                zero = reference.copy()
            restored = np.asarray(get_action(cfg, model, copy.deepcopy(raw), task, **kwargs)).copy()
        if (reference.shape != (8, 7) or not np.isfinite(reference).all()
                or not np.isfinite(zero).all() or not np.array_equal(reference, restored)):
            raise ValueError("Actual-policy shape/finite/saved-reloaded output mismatch")
        if frozen_digest(modules) != before:
            raise ValueError("First-query gate changed frozen model/head/proprio")
    finally:
        if state is not None:
            copy_factors(model, state)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return {"gate": "PASS_B1_TEMPORAL_ACTUAL_MODEL", "version": VERSION,
            "optimization_steps": 1000, "frozen_sha256": before,
            "reload_output_equal": True, "all_parameters_frozen": True,
            "raw_state": np.asarray(raw["state"]).tolist(),
            "state_space": "raw_proprio_before_get_action",
            "single_proprio_normalization_equal": True,
            "proprio_path": "deployed get_action on separate raw copies; one normalization per call",
            "reference_action": reference.tolist(), "zero_branch_action": zero.tolist(),
            "zero_branch_comparison": "Pair with canonical A4 on the exact same initial observation",
            "limits": "Versioned temporal training1000; actual query admission only, no benefit claim"}
