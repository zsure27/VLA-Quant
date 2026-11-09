import json,pathlib,hashlib
root=pathlib.Path.cwd();base=json.load(open('checks/20261109-053-cpu-prep-v1/CPU_RECEIPT.json'));completed={};excluded=[]
profiles=base['profiles'];known={'bf16':None,'awq_w4a16':'awq','smoothquant_w4a8':'smoothquant','smoothquant_w4a4':'smoothquant'}
for p in sorted((root/'runs').glob('*/result.json')):
 r=json.loads(p.read_text());c=r.get('condition')
 if c not in known or r.get('model')!='openvla_oft':continue
 expected=known[c]
 if not r.get('gate','').startswith('PASS') or r.get('checkpoint_receipt_sha256')!=base['checkpoint']['receipt_sha256'] or (expected and r.get('profile_sha256')!=profiles[expected]['sha256']):
  excluded.append({'result':str(p.relative_to(root)),'reason':'gate or exact checkpoint/profile identity not matched'});continue
 key=(c,r['suite'],r['task_id'],r.get('initial_state_index',r.get('reset_index')))
 if key in completed:excluded.append({'result':str(p.relative_to(root)),'reason':'duplicate exact-condition case'});continue
 completed[key]={'result':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert len(completed)==82,len(completed)
catalog=json.load(open('code-cpu-prep-20261109-v1/configs/infra/runtime_commonality_v6/method_catalog.json'));rows=[]
for item in [{'method':'BF16','precision':'BF16'},*catalog['methods']]:
 key='bf16' if item['method']=='BF16' else item['method'].lower()+'_'+item['precision'].lower();pending=[];done=[]
 for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
  for task in range(10):
   for reset in [5,15,25,35,45]:
    case={'suite':suite,'task_id':task,'reset_index':reset};prior=completed.get((key,suite,task,reset))
    if prior:done.append({**case,**prior})
    else:pending.append(case)
 rows.append({'condition':key,'completed':done,'pending':pending,'executable':False,'gpu_run_admitted':False})
result={'scope':'exact checkpoint/profile result inventory; no dispatch; unsupported method bindings stay blocked','completed_unique':len(completed),'duplicates_or_unmatched_excluded':excluded,'conditions':rows}
p=root/'checks/20261109-053-exact-resume-inventory-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print('PASS_EXACT_RESUME_INVENTORY',len(completed),sum(len(x['pending']) for x in rows))
