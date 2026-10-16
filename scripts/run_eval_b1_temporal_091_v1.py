"""Independent temporal80 trained adapter evaluator with matching PTQ numeric bridge."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ["VLA_FROZEN_SOURCE"]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from b1_temporal_eval_contract_091_v1 import MODELS, VERSION, sha, validate_provenance, first_query_contract, matching_ptq_bridge


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(MODELS), required=True)
    parser.add_argument("--model-id", choices=tuple(MODELS.values()), required=True)
    parser.add_argument("--method", choices=("awq",), default="awq")
    parser.add_argument("--weight-bits", type=int, default=2)
    parser.add_argument("--activation-bits", type=int, default=16)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--awq-primary-group64-profile", type=Path, required=True)
    parser.add_argument("--awq-candidate", default="w2-attention-no-clip-primary-g64")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--libero_root", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--initial-state-offset", type=int, default=5)
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=1)
    parser.add_argument("--seed-protocol", choices=("paired",), default="paired")
    parser.add_argument("--trace-actions", action="store_true")
    parser.add_argument("--trace-observations", action="store_true")
    parser.add_argument("--local_log_dir", type=Path, required=True)
    parser.add_argument("--trained-artifact", type=Path, required=True)
    parser.add_argument("--trained-artifact-sha256", required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--recovery-training-manifest", type=Path, required=True)
    parser.add_argument("--training-spec", type=Path, required=True)
    parser.add_argument("--training-pair-gate", type=Path, required=True)
    parser.add_argument("--ptq-reference-sample", type=Path, required=True)
    parser.add_argument("--ptq-reference-sha256", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    record = validate_provenance(args)
    output = args.local_log_dir
    # The locked wrapper creates output/invocation.json before invoking us.
    if (output / "candidate-temporal.json").exists():
        raise ValueError("Candidate output already exists; never repeat a stage")
    output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from qvla.runtime_contract import assert_oft_runtime
    from qvla.reproducibility import seed_all
    from qvla.baseline_contract import checkpoint_identity
    from qvla.extended_peft import descriptors, load_state
    from qvla.model_registry import validate_effective_plan
    from qvla.run_eval_official_quant import load_profiles, apply_quantization
    from qvla.official_quant_adapter import load_official_functions, scope_counts
    from diagnostics.awq_interventions import attention_no_clip_primary_g64_plan
    from diagnostics.probe import initialize_readonly
    assert_oft_runtime()
    seed_all(args.seed, tensorflow=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.libero_root not in sys.path:
        sys.path.insert(0, args.libero_root)
    official = load_official_functions(args.official_root)
    entries, metadata = load_profiles(args.profile, "awq")
    peer = load_profiles([args.awq_primary_group64_profile], "awq")
    plan = attention_no_clip_primary_g64_plan(entries, metadata, peer)
    validate_effective_plan("A4", plan[1])
    identity = checkpoint_identity(args.pretrained_checkpoint)
    if identity != metadata["checkpoint_identity"] or identity != record["checkpoint_identity"]:
        raise ValueError("Actual full checkpoint differs from profile or parent S artifact")
    for name, function in (("awq_auto_scale", "awq_auto_scale_block"),
                           ("awq_auto_clip", "awq_auto_clip"), ("awq_quantizer", "awq_quantize")):
        path = inspect.getsourcefile(inspect.unwrap(official[function]))
        if not path or sha(path) != metadata["official_sources"][name]["sha256"]:
            raise ValueError("Official AWQ implementation changed")
    if scope_counts(list(entries)) != {"total": 422, "primary_vision": 93,
            "fused_vision": 105, "language": 224, "vision": 198, "unexpected": 0}:
        raise ValueError("Quantization connected scope differs from frozen A4")
    # Reuse readonly registration/initialization: checkpoint files are not rewritten.
    _cfg, model, _head, _proprio, _processor = initialize_readonly(args.pretrained_checkpoint, 7)
    if (sha(inspect.getsourcefile(type(model))) != metadata["model_source_sha256"]
            or model.language_model.config._attn_implementation != metadata["llm_attention"]):
        raise ValueError("Actual model class or attention implementation changed")
    rows = descriptors("B3", dict(model.named_modules()), plan[1])
    if rows != record["targets"]:
        raise ValueError("Runtime target descriptors differ from pinned initial factors")
    state = load_state(args.adapter_state, "B3", rows)
    if state is not None and sum(value.numel() for value in state.values()) != 19988480:
        raise ValueError("Runtime rank8 parameter budget changed")
    handles = apply_quantization(model, entries, metadata, "awq", official, 2, 16,
        awq_plan=plan, awq_scope="all", recovery_lora_state=state)
    model.requires_grad_(False)
    model.eval()
    # The evaluator loads the frozen head/proprio itself; discard these diagnostics copies.
    del _head, _proprio, _processor, _cfg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    candidate = {"version": VERSION, "candidate": args.candidate, "model_id": args.model_id,
        "base_model_id": "A4", "optimization_steps": 1000, "training_executed": True,
        "trained_artifact_sha256": sha(args.trained_artifact),
        "adapter_state_sha256": sha(args.adapter_state),
        "trained_state_used": state is not None, "rank": 8 if state is not None else 0,
        "parameters": 19988480 if state is not None else 0,
        "checkpoint_identity": identity, "profiles": {str(p): sha(p) for p in (*args.profile, args.awq_primary_group64_profile)},
        "sources": {str(p): sha(p) for p in
            (Path(__file__), Path(__file__).with_name("b1_temporal_eval_contract_091_v1.py"))},
        "classification": "same-task temporal development; independently trained1000 adapters, no new benefit/independent-replication claim"}
    (output / "candidate-temporal.json").write_text(json.dumps(candidate, indent=2) + "\n")
    import experiments.robot.libero.run_libero_eval as L
    import experiments.robot.robot_utils as R
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from qvla.on_policy_capture import H17SummaryRecorder, OnPolicyObservationCapture
    original_model = R.get_model
    original_libero_model = getattr(L, "get_model", None)
    original_action, original_episode = L.get_action, L.run_episode
    original_argv = list(sys.argv)
    R.get_model = lambda _cfg: model
    if original_libero_model is not None:
        L.get_model = R.get_model
    trace = (output / "policy-queries.jsonl").open("x", encoding="utf-8")
    capture = OnPolicyObservationCapture(output, NUM_ACTIONS_CHUNK)
    h17 = H17SummaryRecorder(model)
    admitted = False

    def traced_episode(*positional, **kwargs):
        task = kwargs.get("task_description")
        if task is None and len(positional) > 2:
            task = positional[2]
        if not isinstance(task, str):
            raise ValueError("Cannot identify rollout task")
        capture.begin_episode(task)
        try:
            result = original_episode(*positional, **kwargs)
        except BaseException:
            capture.end_episode(False, aborted=True)
            raise
        capture.end_episode(bool(result[0]))
        return result

    def traced_action(cfg, actual_model, observation, task, **kwargs):
        nonlocal admitted
        if actual_model is not model:
            raise ValueError("Evaluator did not use the admitted preloaded model")
        if not admitted:
            gate = first_query_contract(model, cfg, observation, task, kwargs, original_action,
                                        record, state, args.adapter_state)
            matching_ptq_bridge(gate, observation, task, model, cfg, args)
            gate["candidate"] = args.candidate
            (output / "actual-model-temporal-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
            admitted = True
        raw_state = np.asarray(observation["state"]).copy()
        h17.reset()
        actions = original_action(cfg, actual_model, observation, task, **kwargs)
        chunk = np.asarray(actions).copy()
        if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
            raise ValueError("Rollout produced malformed or nonfinite action chunk")
        trace.write(json.dumps({"task": task, "state": raw_state.tolist(),
            "state_space": "raw_proprio_before_get_action", "raw_policy_chunk": chunk.tolist(),
            "finite": True, "note": "deployed policy; one normalization before capture"}) + "\n")
        trace.flush()
        capture.record_query(observation, task, chunk, h17.take())
        return actions

    L.run_episode, L.get_action = traced_episode, traced_action
    try:
        sys.argv = [sys.argv[0], "--model_family", "openvla", "--pretrained_checkpoint", args.pretrained_checkpoint,
            "--task_suite_name", args.task_suite_name, "--num_trials_per_task", str(args.num_trials_per_task),
            "--initial_state_offset", str(args.initial_state_offset), "--local_log_dir", str(output),
            "--center_crop", "True", "--seed", str(args.seed), "--env_seed", str(args.env_seed),
            "--seed_protocol", "paired", "--attn_implementation", metadata["llm_attention"]]
        L.eval_libero()
        if not admitted:
            raise ValueError("No actual model query passed the admission gate")
    finally:
        L.run_episode, L.get_action = original_episode, original_action
        R.get_model = original_model
        if original_libero_model is not None:
            L.get_model = original_libero_model
        sys.argv = original_argv
        trace.close()
        h17.close()
        capture.close()
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
