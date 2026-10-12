import json,sys,torch
from pathlib import Path
root=Path('/root/autodl-tmp/qvla-repro/control/b5-pilot-091-20261012-v2');sys.path.insert(0,str(root/'source'))
from qvla.b5_contract import audit_capture,sha,source_hashes,save,require_gpu_admission,require_bank_smoke
from qvla.b5_runtime import Router
base=Path('/root/autodl-tmp/qvla-repro/backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2')
m=json.loads((root/'plans/materials.json').read_text());require_gpu_admission(m);require_bank_smoke(m)
results={};manifests={}
for role in ('router_train','router_dev'):
 d=base/role;mf=audit_capture(d,role);manifests[role]=mf;assert mf['source_sha256']==source_hashes()
 teacher=base/(role+'-teacher');experts=base/(role+'-experts')
 tr=json.loads((teacher/'receipt.json').read_text());er=json.loads((experts/'receipt.json').read_text())
 for folder,receipt,gate in ((teacher,tr,'PASS_B5_TEACHER_CACHE'),(experts,er,'PASS_B5_SAME_OBSERVATION_REGRET')):
  assert receipt['gate']==gate and receipt['role']==role and receipt['capture_sha256']==sha(d/'manifest.json')
  assert receipt['source_sha256']==source_hashes() and receipt['actions_sha256']==sha(folder/'actions.pt')
 assert er['teacher_receipt_sha256']==sha(teacher/'receipt.json')
 ta=torch.load(teacher/'actions.pt',map_location='cpu',weights_only=True);ea=torch.load(experts/'actions.pt',map_location='cpu',weights_only=True)
 assert set(ta)==set(ea)=={r['file'] for r in mf['samples']}
 for row in mf['samples']:
  key=row['file'];assert tr['processed_inputs'][key]==row['processed_inputs']
  assert ta[key].shape==(8,7) and torch.isfinite(ta[key]).all()
  assert ea[key].shape==(2,) and torch.isfinite(ea[key]).all() and (ea[key]>=0).all()
 results[role]={'samples':len(ta),'episodes':len(mf['episodes']),'manifest_sha256':sha(d/'manifest.json'),'expert_receipt_sha256':sha(experts/'receipt.json')}
assert not ({r['observation_sha256'] for r in manifests['router_train']['samples']}&{r['observation_sha256'] for r in manifests['router_dev']['samples']})
for name,task_only in (('router',False),('task-only',True)):
 d=base/name;a=json.loads((d/'artifact.json').read_text())
 assert a['gate']=='PASS_B5_ROUTER_TRAIN_CONTRACT' and a['steps']==500 and a['task_only']==task_only
 assert a['source_sha256']==source_hashes() and a['adapter_sha256']==sha(d/'router.pt')
 assert a['norm_stats_sha256']==sha(Path('/root/autodl-tmp/qvla-repro/control/cpu-audit-20261012/norm-stats-v2.json'))
 for role,prefix in (('router_train','train'),('router_dev','dev')):
  assert a[prefix+'_capture_sha256']==results[role]['manifest_sha256']
  assert a[prefix+'_label_sha256']==results[role]['expert_receipt_sha256']
 model=Router(task_only);state=torch.load(d/'router.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
 assert all(torch.isfinite(v).all() for v in state.values())
 assert sum(p.numel() for p in model.parameters())==a['parameters'];results[name]=a
receipt={'gate':'PASS_B5_CAPTURE_LABEL_FIT_CONTENT_AUDIT','train_dev_observation_intersection':0,'results':results,'source_sha256':source_hashes(),'continuation':'Protocol/content only; no selection by accuracy or success direction'}
save(base/'capture-label-fit-content-gate-v1.json',receipt);print(json.dumps(receipt))

