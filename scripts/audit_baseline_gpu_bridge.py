"""Admit baseline GPU inputs/scope from actual paired traces, never scores."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_extended_peft_eval import audit_case
from qvla.b5_contract import sha,save
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bf16','a0','cpu-receipt','runtime-evaluator','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--count',type=int,default=5);a=p.parse_args()
    cpu=json.loads(a.cpu_receipt.read_text())
    if cpu.get('gate')!='PASS_CPU_ACTUAL_PROFILE_CHECKPOINT_AND_ROW_QUANTIZATION':raise ValueError('Missing actual tensor audit')
    b= audit_case(a.bf16,'BF16',a.offset,a.count);w=audit_case(a.a0,'A0',a.offset,a.count)
    if b['keys']!=w['keys'] or b['first']!=w['first']:raise ValueError('Actual baseline task/reset/first observation pairing differs')
    save(a.output,{'gate':'PASS_GPU_BASELINE_INPUT_AND_QUANTIZATION','checkpoint_sha256':cpu['checkpoint_identity']['sha256'],
        'cpu_receipt_sha256':sha(a.cpu_receipt),'runtime_evaluator_sha256':sha(a.runtime_evaluator),'paired_episodes':len(b['keys']),
        'cases':{name:{'successes':sum(r['success']),'files':r['sha256']} for name,r in [('BF16',b),('A0',w)]},
        'criterion':'hash/command/scope/trace/finite/complete pairing; not target success rate',
        'limits':'Current project baseline admission, not unpublished QVLA AWQ recipe reproduction'})
if __name__=='__main__':main()
