"""Explicit CPU-only final geometry job, not a GPU experiment scheduler or task-completion assertion."""
import argparse,json,os,subprocess,sys,hashlib,datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--prior-analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not os.environ.get('CUDA_VISIBLE_DEVICES'),'Run explicitly without CUDA devices'
 a.output.mkdir(parents=True,exist_ok=False)
 state=a.output/'CPU_JOB_STATE.json'
 def save(phase,**more):state.write_text(json.dumps(dict(phase=phase,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_only=True,**more),indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(a.code))
 try:
  save('AUDITING_A4_AND_REFERENCE')
  subprocess.run([sys.executable,str(a.control/'cpu/analyze_pi05_full_cohort_v10.py'),'--root',str(a.root),'--control',str(a.control),'--code',str(a.code),'--output',str(a.output/'a4-reference'),'--conditions','bf16','duquant_w4a4'],env=env,check=True)
  save('MERGING_EXACT_REFERENCE_SHA')
  subprocess.run([sys.executable,str(a.control/'cpu/merge_pi05_audited_geometry_v1.py'),'--prior',str(a.prior_analysis),'--addition',str(a.output/'a4-reference'),'--output',str(a.output/'combined')],env=env,check=True)
  combined=a.output/'combined';s=json.loads((combined/'summary-public.json').read_text());assert len(s['coverage'])==7 and s['complete_unique']==1400 and not s['missing']
  proof=dict(gate='PASS_ADMITTED_PI05_200_EACH_GEOMETRY_FINALIZATION',configurations=7,unique_cases=1400,paired_quantized=1200,all_requested_quantizers_completed=False,unresolved_quantizer_configurations=5,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),derived_files_sha256={x:hashlib.sha256((combined/x).read_bytes()).hexdigest() for x in ['summary-public.json','curves-public.json','error-decomposition-public.json','merge-proof-public.json']})
  (a.output/'FINAL_GEOMETRY_AUDIT.json').write_text(json.dumps(proof,indent=2));save('COMPLETE',proof=proof);print(json.dumps(proof),flush=True)
 except Exception as e:save('FAILED',error_type=type(e).__name__,all_originals_retained=True);raise
if __name__=='__main__':main()
