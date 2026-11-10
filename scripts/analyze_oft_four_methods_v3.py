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
    hashes={}
    def cached_stream_sha(path):
        path=Path(path);st=path.stat();key=(str(path.resolve()),st.st_size,st.st_mtime_ns)
        if key not in hashes:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
            hashes[key]=h.hexdigest()
        return hashes[key]
    module.sha=cached_stream_sha
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
    extra_figures(out,lookup,module)
    (out/'analysis-source-identity.json').write_text(json.dumps({'frozen_analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frozen_native_audit_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'changes':'artifact path check for GPTQ, exact completed paths, fourth condition and plot dimensions only; streaming cached SHA256 for immutable profile/video files; math and thresholds unchanged','gate':'PASS_AVAILABLE_FOUR_METHOD_RAW_QUERY_ACTION_POSE_VIDEO_AUDIT'},indent=2))


def extra_figures(out,lookup,module):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#2878b5','#e07b39','#b44770','#268c70'];conditions=module.CONDITIONS;suites=module.SUITES;paired={}
    for suite in suites:
        for task in range(10):
            ref=module.load(lookup[suite,task,'bf16'])
            for cond in conditions:
                if (suite,task,cond) not in lookup:continue
                cand=module.load(lookup[suite,task,cond]);t,r,w=module.compare(cand,ref);n=len(t);paired[suite,task,cond]=(cand,ref,cand['position'][:n]-ref['position'][:n],w)
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig=plt.figure(figsize=(18,5))
    for j,suite in enumerate(suites):
        ax=fig.add_subplot(1,4,j+1,projection='3d');ref=module.load(lookup[suite,0,'bf16']);xyz=ref['position']*1000;ax.plot(*xyz.T,color='#444444',lw=2,label='BF16');ax.scatter(*xyz[0],color='black',s=14);all_xyz=[xyz]
        for cond,color in zip(conditions,colors):
            key=suite,0,cond
            if key not in paired:continue
            cand,ref,v,w=paired[key];xyz=cand['position'][:len(v)]*1000;ax.plot(*xyz.T,color=color,label=cond);ax.scatter(*xyz[-1],color=color,s=12);all_xyz.append(xyz)
        ax.set_title(suite+' / task0 / reset5');ax.set_xlabel('Base X (mm)');ax.set_ylabel('Base Y (mm)');ax.set_zlabel('Base Z (mm)');ax.set_box_aspect([1,1,1]);cloud=np.concatenate(all_xyz);center=(cloud.min(0)+cloud.max(0))/2;half=max(float(np.ptp(cloud,axis=0).max()/2),1.0);ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half)
        if j==0:ax.legend(fontsize=6)
    fig.suptitle('Measured end-effector paths; each candidate shown only through its observed paired prefix',y=1.04);save(fig,'05-prespecified-task0-3d-paths')
    fig,axes=plt.subplots(4,10,figsize=(24,10))
    for j,suite in enumerate(suites):
        for task in range(10):
            ax=axes[j,task];ref=module.load(lookup[suite,task,'bf16']);xy=ref['position'][:,:2]*1000;ax.plot(*xy.T,color='#444444',lw=1.4)
            for cond,color in zip(conditions,colors):
                key=suite,task,cond
                if key in paired:
                    cand,ref,v,w=paired[key];xy=cand['position'][:len(v),:2]*1000;ax.plot(*xy.T,color=color,alpha=.8,lw=.8)
            ax.set_aspect('equal',adjustable='datalim');ax.tick_params(labelsize=5);ax.set_title(suite.replace('libero_','')+'/'+str(task),fontsize=7)
    fig.suptitle('All 40 task XY paths (base-frame mm); same initial states, observed paired prefixes',y=1.02);save(fig,'06-all-task-xy-path-coverage')
    fig,axes=plt.subplots(2,4,figsize=(18,9));coverage=[]
    for j,suite in enumerate(suites):
        for cond,color in zip(conditions,colors):
            rows=[paired[suite,t,cond] for t in range(10) if (suite,t,cond) in paired]
            if not rows:continue
            for metric,factor in [(2,1000),(3,180/np.pi)]:
                width=max(len(x[metric]) for x in rows);grid=np.full((len(rows),width,3),np.nan)
                for i,row in enumerate(rows):grid[i,:len(row[metric])]=row[metric]*factor
                count=np.isfinite(grid[:,:,0]).sum(0);mean=np.nanmean(grid,axis=0);bias=np.linalg.norm(mean,axis=1);spread=np.sqrt(np.nanmean(np.sum((grid-mean)**2,axis=2),axis=0));valid=count>=5;bias[~valid]=np.nan;spread[~valid]=np.nan
                ax=axes[metric-2,j];ax.plot(bias,color=color,label=cond+' bias');ax.plot(spread,color=color,ls='--',label=cond+' scatter');coverage.append({'suite':suite,'condition':cond,'metric':'position' if metric==2 else 'world_SO3_log','paired_tasks':len(rows),'minimum_contributors':5,'plotted_steps':int(valid.sum())})
        axes[0,j].set_title(suite);axes[0,j].set_ylabel('Position vector mean / RMS scatter (mm)');axes[1,j].set_ylabel('World SO(3) log mean / RMS scatter (deg)');axes[1,j].set_xlabel('Raw control step; at least 5 observed paired tasks')
    axes[0,0].legend(fontsize=6);fig.suptitle('Solid: norm of across-task mean error vector; dashed: RMS scatter around that vector. Descriptive, not causal.',y=1.02);save(fig,'07-error-vector-bias-versus-scatter')
    (out/'additional-visual-coverage.json').write_text(json.dumps({'scope':'OFT reset5; matched prefixes only; no safety or compensation claim','risk_set_minimum':5,'bias_definition':'norm(mean(error_vector))','scatter_definition':'sqrt(mean(norm(error_vector-mean_vector)^2))','coverage':coverage},indent=2))
    page=out/'index.html';page.write_text(page.read_text()+''.join('<h2>'+n+'</h2><img style="width:100%" src="'+n+'.png">' for n in ['05-prespecified-task0-3d-paths','06-all-task-xy-path-coverage','07-error-vector-bias-versus-scatter']))

if __name__=='__main__':main()
