"""Independent GPTQ trace driver; preserves the frozen native collector."""
from pathlib import Path
import sys

def main():
    root=Path('/root/autodl-tmp/trajectory-drift-20261107')
    code=root/'code-cpu-prep-20261109-v1'
    sys.path[:0]=[str(root/'method-code/20261110-gptq-v2'),str(code),
                 str(root/'sources/openvla-oft'),str(root/'sources/LIBERO')]
    path=code/'qvla/execute_oft_quant_drift_v2.py';body=path.read_text()
    old='from qvla.run_eval_official_quant import load_profiles, apply_quantization'
    assert body.count(old)==1
    body=body.replace(old,'from gptq_oft_v1 import load_profiles, apply_quantization, validate_calibration_actions')
    assert body.count('load_official_functions(args.official_root), 4, args.activation_bits)')==1
    body=body.replace('load_official_functions(args.official_root), 4, args.activation_bits)',
                     'None, 4, args.activation_bits)\n    validate_calibration_actions(model, action_head, proprio_projector, processor, cfg, metadata, args.calibration_dir)')
    body=body.replace("choices=('awq','smoothquant')", "choices=('gptq',)")
    marker="    parser.add_argument('--profile', type=Path, required=True)";assert body.count(marker)==1
    body=body.replace(marker,"    parser.add_argument('--calibration-dir', type=Path, required=True)\n"+marker)
    scope={'__name__':'independent_gptq_trace'}
    exec(compile(body,str(path)+'[GPTQ_ADAPTER]', 'exec'),scope)
    scope['main']()

if __name__=='__main__':main()
