"""Extend frozen measured math to GPTQ and explicit partial-cell coverage."""
import argparse,hashlib,json,sys,types
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();code=root/'code-cpu-prep-20261109-v1';out=a.output.resolve();assert not out.exists()
    sys.path[:0]=[str(code),str(root/'sources/openvla-oft'),str(root/'sources/LIBERO')]
    profile=root/'profiles/20261110-gptq-full-v1/profile.pt';gate=json.loads((profile.parent/'status.json').read_text());assert gate['gate']=='PASS_FULL_422_GPTQ_PACK_AND_RELOAD'
    # Extend the artifact identity check only. All raw query/action/video/pose
    # and matched reset/input checks are exactly the original frozen code.
    native=code/'diagnostics/native_oft_trace_v1.py';body=native.read_text()
    old="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)"
    new="profile = (Path('profiles/20261110-gptq-full-v1/profile.pt') if m['condition'].startswith('gptq') else Path('profiles/20261109-four-suite-reset0-v4')/('awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'))\n        assert result['profile_sha256'] == sha(profile)"
    assert body.count(old)==1;body=body.replace(old,new);name='diagnostics.native_oft_trace_gptq_v1';module=types.ModuleType(name);exec(compile(body,str(native)+'[GPTQ artifact path only]','exec'),module.__dict__);module.CONDITIONS=[*module.CONDITIONS,'gptq_w4a16'];sys.modules[name]=module
    lookup={}
    for f in sorted((root/'runs').glob('*/result.json')):
        row=json.loads(f.read_text())
        if row.get('model')=='openvla_oft' and row.get('gate','').startswith('PASS') and row.get('initial_state_index')==5:lookup.setdefault((row['suite'],row['task_id'],row['condition']),f.parent)
    for control in ['20261110-053-oft-reset5-resume-v1','20261110-053-oft-reset5-resume-v2','20261110-053-gptq-reset5-v1']:
        f=root/'control'/control/'status.json'
        if f.exists():
            for row in json.loads(f.read_text())['completed']:
                case=row['case'];assert hashlib.sha256(Path(row['result']).read_bytes()).hexdigest()==row['sha256'];lookup[case['suite'],case['task_id'],case['condition']]=Path(row['result']).parent
    source=code/'diagnostics/oft_multitask_measured_v1.py';text=source.read_text().replace('from diagnostics.native_oft_trace_v1 import','from diagnostics.native_oft_trace_gptq_v1 import')
    old="cand = ROOT/f'20261109-053-{c}-{s}-task{task}-reset5-v1'";assert text.count(old)==1;text=text.replace(old,"cand = candidates.get((s,task,c),ROOT/f'MISSING-{s}-{task}-{c}')")
    text=text.replace('planned_pairs=120','planned_pairs=160').replace("colors=['#2878b5','#e07b39','#b44770']","colors=['#2878b5','#e07b39','#b44770','#268c70']").replace('np.full((40,3),','np.full((40,len(CONDITIONS)),').replace('range(3));ax.set_xticklabels(CONDITIONS','range(len(CONDITIONS)));ax.set_xticklabels(CONDITIONS')
    ns={'__name__':'measured_gptq_extension','candidates':lookup};exec(compile(text,str(source)+'[four methods and paths]','exec'),ns);ns['reference']=lambda s,t:lookup[s,t,'bf16'];ns['OUT']=out;ns['main']()
    (out/'analysis-source-identity.json').write_text(json.dumps({'frozen_analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frozen_native_audit_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'changes':'artifact path check for GPTQ, exact completed paths, fourth condition and plot dimensions only; math and thresholds unchanged','gate':'PASS_AVAILABLE_FOUR_METHOD_RAW_QUERY_ACTION_POSE_VIDEO_AUDIT'},indent=2))

if __name__=='__main__':main()
