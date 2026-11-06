"""OFT BF16/AWQ/SmoothQuant observer binding, with fail-closed v2 gates.

No new quantizer implementation: reuses the official-source adapters. No GPTQ,
OmniQuant, DuQuant or W3 compatibility is asserted by this entry point.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def validate_admission(identity,gate,argv):
    if gate.get("gate")!="PASS_TRAJECTORY_CAPTURE_ACTUAL_MODEL_V2" or gate.get("identity")!=identity:
        raise ValueError("Actual-model v2 admission missing/mismatched")
    for key in ("same_actions","same_observations","same_seeds","same_query_schedule","finite","frozen","reload"):
        if gate.get(key) is not True:raise ValueError("Observer contract failed: "+key)
    backend=identity.get("backend")
    if identity.get("data_kind")!="measured" or backend not in ("bf16","awq","smoothquant"):
        raise ValueError("Unsupported measured backend")
    if backend=="bf16" and (identity.get("weight_bits")!=16 or identity.get("activation_bits")!=16):
        raise ValueError("BF16 identity must be W16A16")
    def option(name):
        if argv.count(name)!=1 or argv.index(name)+1>=len(argv):raise ValueError("Missing/duplicate argv: "+name)
        return argv[argv.index(name)+1]
    if option("--pretrained_checkpoint")!=identity["checkpoint"]:raise ValueError("Checkpoint argument mismatch")
    if option("--seed_protocol" if backend=="bf16" else "--seed-protocol")!="paired":raise ValueError("Paired protocol required")
    if backend!="bf16":
        if option("--method")!=backend or int(option("--weight-bits"))!=identity["weight_bits"] or int(option("--activation-bits"))!=identity["activation_bits"]:
            raise ValueError("Quantizer identity mismatch")
        actual=[]
        for i,value in enumerate(argv):
            if value=="--profile": actual.append(argv[i+1])
        if not actual or len(actual)!=len(set(actual)) or set(actual)!=set(identity.get("profile_sha256",{})):raise ValueError("Profile identity mismatch")
    if backend=="awq" and (identity["activation_bits"]!=16 or identity["weight_bits"] not in (2,4,8)):
        raise ValueError("AWQ supports weight-only W2/W4/W8 A16; hybrid must have a distinct backend")
    if backend=="smoothquant":
        if identity["weight_bits"] not in (4,8,16) or identity["activation_bits"] not in (4,8,16):raise ValueError("Unsupported SQ precision")
        c0=gate.get("smooth_only_equivalence",{})
        if not identity.get("coordinate_id") or c0.get("gate")!="PASS_SQ_SMOOTH_ONLY_EQUIVALENCE_V2" or c0.get("coordinate_id")!=identity.get("coordinate_id") or c0.get("passed") is not True:
            raise ValueError("SQ smooth-only equivalence must pass before activation studies")
    return backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity",type=Path,required=True)
    p.add_argument("--admission",type=Path,required=True)
    p.add_argument("--capture-root",type=Path,required=True)
    p.add_argument("evaluator_argv",nargs=argparse.REMAINDER)
    args=p.parse_args()
    identity=json.loads(args.identity.read_text(encoding="utf-8"));gate=json.loads(args.admission.read_text(encoding="utf-8"))
    argv=args.evaluator_argv[1:] if args.evaluator_argv[:1]==["--"] else args.evaluator_argv
    backend=validate_admission(identity,gate,argv)
    for name in ("trajectory_capture_v1.py","run_trajectory_diagnostic_v2.py"):
        path=Path(__file__).with_name(name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=gate.get("source_sha256",{}).get(name):raise ValueError("Unpinned source: "+name)
    for path,sha in identity.get("profile_sha256",{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError("Profile content changed")
    from qvla.baseline_contract import checkpoint_identity
    if checkpoint_identity(identity["checkpoint"])["sha256"]!=identity["checkpoint_sha256"]:raise ValueError("Checkpoint content changed")
    from qvla.runtime_contract import assert_oft_runtime
    assert_oft_runtime()
    from qvla.trajectory_capture_v1 import observe_episode
    import experiments.robot.libero.run_libero_eval as evaluator
    saved=sys.argv
    try:
        sys.argv=[sys.argv[0]]+argv
        with observe_episode(evaluator,args.capture_root,identity):
            if backend=="bf16":evaluator.eval_libero()
            else:
                from qvla.run_eval_official_quant import main as quant_main
                quant_main()
    finally:sys.argv=saved


if __name__=="__main__":main()
