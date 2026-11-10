"""Reuse frozen measured-analysis math; change only exact case path resolution."""
import argparse,hashlib,json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);a=p.parse_args();root=a.root.resolve();c=a.control.resolve();code=root/'code-cpu-prep-20261109-v1'
    status=json.loads((c/'status.json').read_text());assert status['status']=='COMPLETE_PENDING_PAIRED_AUDIT_AND_CLOSURE' and len(status['completed'])==78
    lookup={}
    for f in sorted((root/'runs').glob('*/result.json')):
        r=json.loads(f.read_text())
        if r.get('model')=='openvla_oft' and r.get('gate','').startswith('PASS') and r.get('initial_state_index')==5:
            key=r['suite'],r['task_id'],r['condition']
            lookup.setdefault(key,f.parent)
    for row in status['completed']:
        case=row['case'];lookup[case['suite'],case['task_id'],case['condition']]=Path(row['result']).parent
    suites=['libero_spatial','libero_object','libero_goal','libero_10'];conditions=['awq_w4a16','smoothquant_w4a8','smoothquant_w4a4'];assert all((s,t,k) in lookup for s in suites for t in range(10) for k in ['bf16',*conditions])
    sys.path[:0]=[str(code),str(root/'sources/openvla-oft'),str(root/'sources/LIBERO')]
    source=code/'diagnostics/oft_multitask_measured_v1.py';text=source.read_text();old="cand = ROOT/f'20261109-053-{c}-{s}-task{task}-reset5-v1'";assert text.count(old)==1;text=text.replace(old,'cand = candidates[s,task,c]')
    out=root/'analysis/20261110-oft-reset5-complete-v1';assert not out.exists()
    ns={'__name__':'measured_analysis_resume','candidates':lookup};exec(compile(text,str(source),'exec'),ns);ns['reference']=lambda s,t:lookup[s,t,'bf16'];ns['OUT']=out;ns['main']()
    report=json.loads((out/'summary.json').read_text());assert report['available_pairs']==120 and not report['missing']
    (out/'analysis-source-identity.json').write_text(json.dumps({'frozen_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'path_resolution_only_patch':old,'round_plan_sha256':hashlib.sha256((c/'plan.json').read_bytes()).hexdigest(),'gate':'PASS_120_PAIRED_RAW_QUERY_ACTION_POSE_VIDEO_AUDIT'},indent=2))

if __name__=='__main__':main()
