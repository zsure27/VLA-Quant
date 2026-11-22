"""Combine completed same-backend cohorts only when every reused BF16 result SHA agrees."""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--addition',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 def read(folder,name):return json.loads((folder/name).read_text(encoding='utf-8'))
 def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 left=read(a.prior,'summary-public.json');right=read(a.addition,'summary-public.json')
 for s in [left,right]:
  assert s['gate']=='PASS_COMPLETE_200_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT' and not s['missing']
  assert s['native_jax_combined'] is False
  assert all(x['complete']==200 for x in s['coverage'].values())
 for key in ['backend','reset_indices','suites','development_tasks']:assert left[key]==right[key],key
 assert set(left['coverage'])&set(right['coverage'])=={'bf16'}
 def casekey(row):return row['condition'],row['suite'],row['task_id'],row['reset_index']
 bfleft={casekey(x):x for x in left['records'] if x['condition']=='bf16'};bfright={casekey(x):x for x in right['records'] if x['condition']=='bf16'}
 assert bfleft==bfright and len(bfleft)==200
 la=read(a.prior,'audit-private.json')['cases'];ra=read(a.addition,'audit-private.json')['cases']
 lm={x['result']:x for x in la};rm={x['result']:x for x in ra};assert len(lm)==len(la) and len(rm)==len(ra)
 common=set(lm)&set(rm);assert len(common)==200
 for path in common:
  assert lm[path]['result_sha256']==rm[path]['result_sha256']
  assert lm[path]['audit']['plan_sha256']==rm[path]['audit']['plan_sha256']
  assert lm[path]['audit']['result_sha256']==rm[path]['audit']['result_sha256']
 records=left['records']+[x for x in right['records'] if x['condition']!='bf16'];assert len({casekey(x) for x in records})==len(records)
 summary=dict(left,coverage={**left['coverage'],**right['coverage']},records=records,pairs=left['pairs']+right['pairs'],complete_unique=len(records),paired_quantized=len(left['pairs'])+len(right['pairs']))
 assert len(summary['coverage'])==7 and len(records)==1400 and len(summary['pairs'])==1200
 curves=read(a.prior,'curves-public.json')+read(a.addition,'curves-public.json');assert len(curves)==24 and len({(x['condition'],x['suite']) for x in curves})==24
 ld=read(a.prior,'error-decomposition-public.json');rd=read(a.addition,'error-decomposition-public.json');assert ld['gate']==rd['gate']=='PASS_PAIRED_VECTOR_CENTER_SPREAD_RMS_IDENTITY'
 decomposed=dict(ld,curves=ld['curves']+rd['curves']);assert len(decomposed['curves'])==24
 examples=read(a.prior,'examples-private.json')+[x for x in read(a.addition,'examples-private.json') if x['condition']!='bf16'];assert len(examples)==28
 proof=dict(gate='PASS_SEVEN_COMPLETE_CONFIGURATIONS_EXACT_REFERENCE_SHA_REUSE',reference_cases_sha_verified=200,unique_cases=1400,paired_quantized_cases=1200,first_six_conditions_not_recollected=True,native_jax_combined=False,input_hashes={name:{f:digest(folder/f) for f in ['summary-public.json','curves-public.json','error-decomposition-public.json','audit-private.json']} for name,folder in [('prior',a.prior),('addition',a.addition)]})
 a.output.mkdir(parents=True,exist_ok=False)
 for name,value in [('summary-public.json',summary),('curves-public.json',curves),('error-decomposition-public.json',decomposed),('examples-private.json',examples),('audit-private.json',dict(cases=la+[x for x in ra if x['result'] not in common])),('merge-proof-public.json',proof)]:
  (a.output/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
