"""Exercise the actual resume admission block, without loading a policy/GPU."""
import copy, hashlib, json, math, tempfile, types
from pathlib import Path

source=Path(__file__).with_name('calibrate_pi05_omni_lwc_resume_v2.py').read_text(encoding='utf-8')
block=source.split('# Resume only completed immutable independent units, never an unfinished optimizer.\n',1)[1].split('\ndef action(policy,row):',1)[0]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trial(change=None):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);parent=root/'parent';output=root/'output'
        for p in [parent/'units',output/'units']:p.mkdir(parents=True)
        adapter=root/'adapter.py';adapter.write_text('frozen teacher source')
        artifact=dict(epochs=20,learning_rate=.01,weight_decay=0,LET=False,disabled_quant_bitwise=True,gradient_finite_nonzero=True,teacher_unchanged=True,native_unit_reconstruction=True,calls=32,weights={'q_proj':{}},loss_first=1.,loss_last=.5)
        file=parent/'units/000.pt';file.write_text(json.dumps(artifact))
        payload=dict(backend='test',epochs=20,entries=[],full_policy_admitted=False)
        prior=copy.deepcopy(payload);prior['entries']=[dict(name='block.self_attn',file='units/000.pt',sha256=digest(file),calls=32,targets=1,loss_first=1.,loss_last=.5)]
        if change:change(prior,artifact,file)
        (parent/'progress.json').write_text(json.dumps(prior))
        context=dict(a=types.SimpleNamespace(resume_from=parent,output=output),json=json,payload=payload,digest=digest,Path=Path,cpu=dict(adapter_sha256=digest(adapter)),native_adapter=types.SimpleNamespace(__file__=str(adapter)),torch=types.SimpleNamespace(load=lambda path,**kw:json.loads(Path(path).read_text())),np=types.SimpleNamespace(isfinite=math.isfinite),write_json=lambda p,v:Path(p).write_text(json.dumps(v)))
        exec(block,context)
        assert len(payload['entries'])==1 and (output/'units/000.pt').read_bytes()==file.read_bytes()
        assert (parent/'progress.json').read_text()==json.dumps(prior)

trial()
bad=[lambda p,a,f:p.update(backend='different'),lambda p,a,f:p['entries'][0].update(sha256='0'*64),lambda p,a,f:p['entries'][0].update(file='../outside.pt'),lambda p,a,f:p.update(full_policy_admitted=True),lambda p,a,f:p.update(entries=[])]
def wrong_epoch(p,a,f):
    a['epochs']=2;f.write_text(json.dumps(a));p['entries'][0]['sha256']=digest(f)
bad.append(wrong_epoch)
for change in bad:
    try:trial(change)
    except AssertionError:pass
    else:raise RuntimeError('Invalid resume accepted')
print(json.dumps(dict(gate='PASS_OMNI_RESUME_ACTUAL_PROVENANCE_BLOCK_CPU',valid_prefix_accepted=True,invalid_cases_rejected=len(bad),gpu_model_test=False)))
