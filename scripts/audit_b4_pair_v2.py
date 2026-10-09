"""Fresh B3/B4 paired development gates, with explicitly archival reference metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from qvla.split_scope import sha
from qvla.paired_metrics import comparison
from scripts.audit_extended_peft_eval import audit_case
from scripts.audit_b4_data_scope import verify_receipt, REFERENCES

CASES=('B3','B4')
PHASES={'micro':(20,1),'first50':(20,5),'next50':(25,5),'last100':(30,10)}


def audit(root,phase):
    receipt=verify_receipt(root/'data-scope-receipt.json')
    offset,count=PHASES[phase]
    results={c:audit_case(root/f'eval-{phase}'/c,c,offset,count) for c in CASES}
    base=results['B3']
    if results['B4']['keys']!=base['keys'] or results['B4']['first']!=base['first']:
        raise ValueError('Fresh pair differs in keys/first policy-visible observation')
    reference={(r['task_id'],r['init_state_index']):r for r in receipt['reference_rows']}
    training=json.loads(Path(receipt['training_manifest']).read_text(encoding='utf-8'))['samples']
    train_obs={r['observation_sha256'] for r in training}
    for case in CASES:
        events=[json.loads(line) for line in (root/f'eval-{phase}'/case/'on-policy-events.jsonl').read_text(encoding='utf-8').splitlines()]
        if train_obs & {r['observation_sha256'] for r in events if r['record_type']=='query'}:
            raise ValueError('Fresh evaluation contains a training observation')
    rows=[]
    for serial,key in enumerate(base['keys']):
        old=reference[(key[0],key[1])]
        if list(key)!=old['key'] or base['first'][serial]!=old['first']:
            raise ValueError('Archived reference keys/first policy input differ; rerun references under a new plan')
        rows.append({'task_id':key[0],'init_state_index':key[1],
                     **{c:results[c]['success'][serial] for c in CASES},**{c:old[c] for c in REFERENCES}})
    return {'gate':f'PASS_PROTOCOL_B4_PAIR_V2_{phase.upper()}','root':str(root),'phase':phase,
            'paired_rows':rows,'paired_episodes':len(rows),'data_scope_sha256':sha(root/'data-scope-receipt.json'),
            'cases':{c:{'successes':sum(r['success']),'episodes':len(rows),'sha256':r['sha256']} for c,r in results.items()},
            'primary':comparison(rows,'B4','B3'),
            'archival_references':{c:comparison(rows,'B4',c) for c in REFERENCES},
            'classification':'within-task reused development; B3/B4 fresh, A4/BF16/A0 archival matched inputs',
            'reference_limit':'No claim of a concurrent rerun or independent reproduction of reference trajectories',
            'continue_rule':'protocol only, never success direction'}


def require_prior(root,phase):
    record=json.loads((root/f'{phase}-protocol-gate.json').read_text(encoding='utf-8'))
    if record!=audit(root,phase): raise ValueError('Prior gate/actual data changed')


def combined(root):
    rows=[]
    for phase in ('first50','next50','last100'):
        require_prior(root,phase)
        rows.extend(json.loads((root/f'{phase}-protocol-gate.json').read_text(encoding='utf-8'))['paired_rows'])
    if len(rows)!=200 or {(r['task_id'],r['init_state_index']) for r in rows}!={(t,r) for t in range(10) for r in range(20,40)}:
        raise ValueError('Require exactly 200 non-overlapping paired episodes; exclude micro duplicates')
    primary=comparison(rows,'B4','B3')
    return {'gate':'PASS_PROTOCOL_B4_PAIR_V2_COMBINED200','classification':'within-task development only',
            'paired_rows':rows,'primary':primary,'successes':{c:sum(r[c] for r in rows) for c in (*CASES,*REFERENCES)},
            'archival_references':{c:comparison(rows,'B4',c) for c in REFERENCES},
            'failure_pairs':[r for r in rows if not r['B3'] or not r['B4']],
            'decision':'SUPPORTED_DEVELOPMENT_INCREMENT' if primary['task_cluster_bootstrap_95ci_fraction'][0]>0
                       else 'INCONCLUSIVE_INCREMENT' if primary['net']>0 else 'NO_POSITIVE_INCREMENT',
            'reference_limit':'Reference outcomes reused once, not concurrent reruns or independent samples',
            'locked':['Router','P4','P5','offline_final_holdout']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=(*PHASES,'combined'))
    p.add_argument('--output',type=Path);p.add_argument('--require-prior',choices=tuple(PHASES))
    a=p.parse_args()
    if a.require_prior:
        require_prior(a.root,a.require_prior);print('PASS_PRIOR_PROTOCOL');return
    if not a.phase or not a.output:p.error('--phase and --output required')
    if a.output.exists():raise ValueError('Never overwrite a gate')
    record=combined(a.root) if a.phase=='combined' else audit(a.root,a.phase)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'gate':record['gate'],'primary':record['primary']}))


if __name__=='__main__':main()
