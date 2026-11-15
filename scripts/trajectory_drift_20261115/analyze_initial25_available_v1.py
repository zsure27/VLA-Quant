"""Terminal/explicit quota-stop initial25 analysis; frozen geometry and thresholds.

Only derived metrics/plots are public; provenance and original traces stay private.
No missing trajectory padding, step alignment, or success-driven case selection.
"""
import argparse,hashlib,json,os,sys,types
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve();control=a.control.resolve();a.output.mkdir(exist_ok=False,parents=True);os.environ['CUDA_VISIBLE_DEVICES']='';os.chdir(r)
 sys.path[:0]=[str(r/'code-cpu-prep-20261109-v1'),str(r/'sources/openvla-oft'),str(r/'sources/LIBERO')]
 from qvla.infra.common import digest
 from qvla.trajectory_diagnostics_v1 import sustained_windows
 series=json.loads((control/'pending-series.json').read_text());profiles={k:v['sha256'] for k,v in series['profiles'].items()};conditions=list(profiles);suites=['libero_spatial','libero_object','libero_goal','libero_10']
 source=r/'code-cpu-prep-20261109-v1/diagnostics/native_oft_trace_v1.py';assert digest(source)=='b084a872cccbeb839ac83bc74cc88fb93ce0cfa841d616b72addc21b498a576f'
 old="profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)";body=source.read_text();assert body.count(old)==1;body=body.replace(old,"assert result['profile_sha256'] == expected_profiles[m['condition']]");m=types.ModuleType('_frozen_initial25_geometry');m.expected_profiles=profiles;exec(compile(body,str(source)+'[profile dispatch only]','exec'),m.__dict__)
 hashes={}
 def cached(path):
  path=Path(path);stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
  if key not in hashes:hashes[key]=digest(path)
  return hashes[key]
 m.sha=cached;references={};rows=[];traces={};missing=[]
 for case in series['cases']:references[(case['suite'],case['task_id'])]=Path(case['reference_result'])
 for stage in sorted((control/'stages').iterdir()):
  if not (stage/'status.json').exists():continue
  status=json.loads((stage/'status.json').read_text());assert status['state'] not in ['RUNNING','PREFLIGHT'], 'Stop active collection before analysis'
  for file in stage.glob('*-libero*-cases.json'):
   for record in json.loads(file.read_text())['completed']:
    case=record['case'];assert record['first_input_paired'] is True;assert cached(record['result_path'])==record['result_sha256'];assert cached(case['reference_result'])==case['reference_sha256']
    cand=m.load(Path(record['result_path']).parent);bf=m.load(Path(case['reference_result']).parent);t,rot,_=m.compare(cand,bf);assert np.isfinite(t).all() and np.isfinite(rot).all();n=len(t)
    row=dict(condition=case['condition'],suite=case['suite'],task_id=case['task_id'],reset_index=25,success=bool(cand['result']['success']),reference_success=bool(bf['result']['success']),common_steps=n,candidate_steps=len(cand['position']),reference_steps=len(bf['position']),translation_p95_mm=float(np.percentile(t*1000,95)),rotation_p95_deg=float(np.percentile(rot,95)),translation_windows=sustained_windows(t,.005,5),rotation_windows=sustained_windows(rot,2.,5),wall_seconds=record['wall_seconds'])
    delta=cand['first_chunk']-bf['first_chunk'];assert delta.shape==(8,7) and np.isfinite(delta).all();k=min(n,8);row['first_chunk_translation_rmse_by_position']=np.sqrt(np.mean(delta[:k,:3]**2,axis=1)).tolist();row['first_chunk_rotation_rmse_by_position']=np.sqrt(np.mean(delta[:k,3:6]**2,axis=1)).tolist()
    key=(row['condition'],row['suite'],row['task_id']);assert key not in traces;traces[key]=dict(t=t*1000,rot=rot,vector=(cand['position'][:n]-bf['position'][:n])*1000);rows.append(row)
 coverage={}
 for c in conditions:
  available=[x for x in rows if x['condition']==c];coverage[c]=dict(completed=len(available),planned=40,success=sum(x['success'] for x in available),reference_success=sum(x['reference_success'] for x in available),rescue=sum(x['success'] and not x['reference_success'] for x in available),breaks=sum(not x['success'] and x['reference_success'] for x in available),coverage_complete=len(available)==40)
  missing.extend(dict(condition=c,suite=s,task_id=i,reset_index=25) for s in suites for i in range(10) if (c,s,i) not in traces)
 reference_results=[json.loads(path.read_text()) for path in references.values()]
 result=dict(reference_cohort=dict(completed=len(reference_results),success=sum(bool(x['success']) for x in reference_results),reset_index=25),window_index_base=0,gate='PASS_INITIAL25_AVAILABLE_TERMINAL_DERIVED_ANALYSIS',initial_states=[25],development_tasks=40,paired_trajectories=len(rows),conditions=conditions,coverage=coverage,missing=missing,metrics=rows,thresholds=dict(translation_mm=5,rotation_deg=2,persistence_steps=5,source='Previously frozen BF16 repeat controls; never refit to candidates'),qualification='Initial25 only. Unequal coverage is not a fair ranking; closed-loop separation includes feedback. Threshold windows are not universal changepoints or OPD efficacy.',original_observations_actions_included=False)
 (a.output/'summary-public.json').write_text(json.dumps(result,indent=2)+'\n');plot(result,traces,a.output,suites)
 print(json.dumps(dict(pairs=len(rows),coverage=coverage)))

def plot(data,traces,out,suites):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 conditions=data['conditions'];colors=dict(zip(conditions,plt.cm.tab10.colors));plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(4,4,figsize=(17,12),gridspec_kw={'height_ratios':[3,3,1,1]},constrained_layout=True)
 for col,s in enumerate(suites):
  for c in conditions:
   rows=[x for x in data['metrics'] if x['suite']==s and x['condition']==c]
   for ai,key in enumerate(['t','rot']):
    values=[traces[c,s,row['task_id']][key] for row in rows]
    if not values:continue
    grid=np.full((len(values),max(map(len,values))),np.nan)
    for i,x in enumerate(values):grid[i,:len(x)]=x
    count=np.isfinite(grid).sum(0);valid=count>=5;step=np.arange(1,grid.shape[1]+1);median=np.nanmedian(grid,0);lo,hi=np.nanpercentile(grid,[25,75],axis=0);axes[ai,col].plot(step,np.where(valid,median,np.nan),color=colors[c],label=c);axes[ai,col].fill_between(step,lo,hi,where=valid,color=colors[c],alpha=.10);axes[ai+2,col].step(step,count,color=colors[c],where='post')
  for ai in range(4):axes[ai,col].grid(alpha=.15);axes[ai,col].set_ylim(bottom=0)
  axes[0,col].set_title(s+' | initial25 / up to10 tasks');axes[0,col].set_ylabel('Position separation (mm)');axes[1,col].set_ylabel('SO(3) separation (deg)');axes[2,col].set_ylabel('Position pairs');axes[3,col].set_ylabel('Rotation pairs');axes[3,col].set_xlabel('Executed environment step');axes[0,col].axhline(5,color='gray',ls=':');axes[1,col].axhline(2,color='gray',ls=':')
 axes[0,0].legend(fontsize=6);fig.suptitle('OpenVLA-OFT initial25 | median/IQR only while >=5 distinct tasks remain\nAvailable common prefixes; no padding; risk counts shown. Frozen thresholds are not safety limits.');fig.savefig(out/'01-drift-risk.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
 for col,s in enumerate(suites):
  for c in conditions:
   rows=[x for x in data['metrics'] if x['suite']==s and x['condition']==c]
   for ai,key in enumerate(['first_chunk_translation_rmse_by_position','first_chunk_rotation_rmse_by_position']):
    if not rows:continue
    grid=np.full((len(rows),8),np.nan)
    for i,row in enumerate(rows):grid[i,:len(row[key])]=row[key]
    med=np.nanmedian(grid,0);lo,hi=np.nanpercentile(grid,[25,75],axis=0);axes[ai,col].plot(range(1,9),med,color=colors[c],label=c);axes[ai,col].fill_between(range(1,9),lo,hi,color=colors[c],alpha=.10)
  for ai in range(2):axes[ai,col].grid(alpha=.15);axes[ai,col].set_ylim(bottom=0);axes[ai,col].set_xlabel('Position in first executed8-action chunk')
  axes[0,col].set_title(s);axes[0,col].set_ylabel('Translation command RMSE');axes[1,col].set_ylabel('Rotation command RMSE')
 axes[0,0].legend(fontsize=6);fig.suptitle('Exactly same first observation | native command units, not mm / pose degrees');fig.savefig(out/'02-same-input-chunk.png',dpi=150);plt.close(fig)
 keys=[(s,i) for s in suites for i in range(10)];lookup={(x['suite'],x['task_id'],x['condition']):x for x in data['metrics']};fig,axes=plt.subplots(1,3,figsize=(16,13),constrained_layout=True);cmap=plt.cm.viridis.copy();cmap.set_bad('#dddddd')
 for ax,key,title in [(axes[0],'translation_p95_mm','log10(1+position P95 mm)'),(axes[1],'rotation_p95_deg','log10(1+SO(3) P95 degrees)'),(axes[2],'success','Success:0/1; missing=grey')]:
  grid=np.array([[lookup.get((s,i,c),{}).get(key,np.nan) for c in conditions] for s,i in keys],dtype=float)
  if key!='success':grid=np.log10(1+grid)
  im=ax.imshow(grid,aspect='auto',cmap=cmap,interpolation='nearest');ax.set_xticks(range(len(conditions)));ax.set_xticklabels(conditions,rotation=60,ha='right',fontsize=7);ax.set_yticks(range(40));ax.set_yticklabels([f'{s}/{i}' for s,i in keys],fontsize=6);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.5)
 fig.suptitle('Initial25 available cases |40 development tasks | missing cases never filled as failures');fig.savefig(out/'03-case-map.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
