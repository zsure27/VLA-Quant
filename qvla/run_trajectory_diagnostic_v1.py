"""Versioned infra command binding. Requires separately verified GPU admission."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["bf16", "awq"], required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("evaluator_argv", nargs=argparse.REMAINDER)
    args = p.parse_args()
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    gate = json.loads(args.admission.read_text(encoding="utf-8"))
    capture_source = Path(__file__).with_name("trajectory_capture_v1.py")
    if (gate.get("gate") != "PASS_TRAJECTORY_CAPTURE_ACTUAL_MODEL_V1" or
        gate.get("identity") != identity or
        gate.get("capture_source_sha256") != hashlib.sha256(capture_source.read_bytes()).hexdigest() or
        not all(gate.get(k) is True for k in ("same_actions", "same_observations", "same_seeds",
                                              "same_query_schedule", "finite", "frozen", "reload"))):
        raise ValueError("Actual-model observer admission missing or mismatched")
    if identity.get("data_kind") != "measured" or identity.get("backend") != args.backend:
        raise ValueError("Measured identity/backend mismatch")
    argv = args.evaluator_argv
    if argv and argv[0] == "--":
        argv = argv[1:]
    def option(name):
        if argv.count(name)!=1: raise ValueError("Exactly one option required: "+name)
        return argv[argv.index(name)+1]
    if option("--pretrained_checkpoint")!=identity["checkpoint"]:
        raise ValueError("Actual checkpoint argument differs from admitted identity")
    protocol="--seed_protocol" if args.backend=="bf16" else "--seed-protocol"
    if option(protocol)!="paired": raise ValueError("Paired seed protocol required")
    if args.backend=="awq" and (option("--method")!="awq" or int(option("--weight-bits"))!=identity["weight_bits"] or int(option("--activation-bits"))!=16):
        raise ValueError("Actual quantization arguments differ from identity")
    from qvla.baseline_contract import checkpoint_identity
    if checkpoint_identity(identity["checkpoint"])["sha256"]!=identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint content changed after admission")
    from qvla.runtime_contract import assert_oft_runtime
    assert_oft_runtime()
    import experiments.robot.libero.run_libero_eval as evaluator
    from qvla.trajectory_capture_v1 import observe_episode
    saved = sys.argv
    try:
        sys.argv = [sys.argv[0]] + argv
        with observe_episode(evaluator, args.capture_root, identity):
            if args.backend == "bf16":
                evaluator.eval_libero()
            else:
                from qvla.run_eval_official_quant import main as quant_main
                quant_main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
