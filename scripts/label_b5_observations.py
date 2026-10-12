"""Frozen teacher/expert labels on TRAIN/DEV current observations; no eval input."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import audit_capture,save,sha,require_gpu_admission,require_bank_smoke,source_hashes,VERSION
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--materials',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--role',choices=('router_train','router_dev'),required=True);p.add_argument('--phase',choices=('teacher','experts'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher',type=Path);a=p.parse_args()
    m=json.loads(a.materials.read_text());require_gpu_admission(m);require_bank_smoke(m);manifest=audit_capture(a.capture,a.role)
    if a.output.exists():raise ValueError('Never overwrite label output')
    for key in ('oft_root','libero_root'):sys.path.insert(0,m[key])
    import torch
    from qvla.b5_runtime import initialize,normalized_query
    a.output.mkdir(parents=True);ctx=initialize(m,teacher=a.phase=='teacher');values={};hashes={}
    prior=None
    if a.phase=='experts':
        if a.teacher is None:raise ValueError('Frozen same-observation teacher cache required')
        receipt=json.loads((a.teacher/'receipt.json').read_text())
        if receipt.get('gate')!='PASS_B5_TEACHER_CACHE' or receipt['capture_sha256']!=sha(a.capture/'manifest.json') or receipt['source_sha256']!=source_hashes() or sha(a.teacher/'actions.pt')!=receipt['actions_sha256']:raise ValueError('Teacher cache identity mismatch')
        prior=torch.load(a.teacher/'actions.pt',map_location='cpu',weights_only=True)
    for row in manifest['samples']:
        path=a.capture/row['file'];sample=row['file']
        if a.phase=='teacher':
            value,h=normalized_query(path,*ctx[:5],ctx[5],ctx[6],0)
            if h!=row['processed_inputs']:raise ValueError('Teacher replay differs from actual collector processed inputs')
            values[sample]=value;hashes[sample]=h
        else:
            targets=[]
            for expert in (0,1):
                value,h=normalized_query(path,*ctx[:5],ctx[5],ctx[6],expert)
                if h!=receipt['processed_inputs'][sample]:raise ValueError('Teacher/student processed observation differs')
                targets.append(value)
            errors=torch.stack([torch.nn.functional.smooth_l1_loss(x,prior[sample],beta=.1) for x in targets])
            values[sample]=errors
    torch.save(values,a.output/'actions.pt')
    save(a.output/'receipt.json',{'version':VERSION,'gate':'PASS_B5_TEACHER_CACHE' if a.phase=='teacher' else 'PASS_B5_SAME_OBSERVATION_REGRET',
        'role':a.role,'capture_sha256':sha(a.capture/'manifest.json'),'actions_sha256':sha(a.output/'actions.pt'),
        'teacher_receipt_sha256':None if a.teacher is None else sha(a.teacher/'receipt.json'),
        'processed_inputs':hashes,'source_sha256':source_hashes(),'samples':len(values),'loss':'normalized 8x7 SmoothL1 beta0.1; lower is better; ties choose E1'})
    if ctx[5] is not None:ctx[5].close()
    for h in ctx[-1]:h.remove()
if __name__=='__main__':main()
