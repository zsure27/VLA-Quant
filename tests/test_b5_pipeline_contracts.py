"""CPU tests of grouped data leakage rejection and the real selector CLI."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from qvla.b5_contract import VERSION,audit_capture,check_range,sha,source_hashes
from qvla.b5_features import CausalFeatureEncoder
from qvla.b5_features import normalized_proprio,RAW_STATE
from qvla.on_policy_capture import observation_hash
ROOT=Path(__file__).resolve().parents[1]
def write(path,value):path.write_text(json.dumps(value))
def fixture(root,role,resets):
    root.mkdir();rows=[];norm={'q01':[-1.]*8,'q99':[1.]*8,'mask':[True]*8}
    for e in (0,1):
        for t in range(10):
            for reset in resets:
                name=f'e{e}-t{t}-r{reset}.npz'
                # Entire test fixture is synthetic, kept in pytest's ignored temp.
                image=np.full((8,8,3),t*20+reset,np.uint8);wrist=np.full((8,8,3),e*100,np.uint8)
                np.savez(root/name,image=image,wrist_image=wrist,router_image=image,router_wrist_image=wrist,state=np.full(8,.1*reset),instruction=f'task {t}',action=np.zeros(7))
                with np.load(root/name) as s:obs=observation_hash(s['image'],s['wrist_image'],normalized_proprio(s['state'],state_space=RAW_STATE,norm_stats=norm),str(s['instruction'].item()))
                rows.append({'file':name,'sha256':sha(root/name),'observation_sha256':obs,'collector_expert':e,'task_id':t,'init_state_index':reset,'query_in_episode':0})
    write(root/'manifest.json',{'version':VERSION,'role':role,'state_space':'raw_proprio_before_get_action','proprio_norm_stats':norm,'samples':rows})
    return rows
def labels(path,capture,rows):
    path.mkdir();torch.save({r['file']:torch.tensor([.1,.5]) if r['task_id']%2 else torch.tensor([.5,.1]) for r in rows},path/'actions.pt')
    write(path/'receipt.json',{'gate':'PASS_B5_SAME_OBSERVATION_REGRET','role':json.loads((capture/'manifest.json').read_text())['role'],'capture_sha256':sha(capture/'manifest.json'),'source_sha256':source_hashes(),'actions_sha256':sha(path/'actions.pt')})
def test_grouped_capture_rejects_reset_or_content_leakage(tmp_path):
    path=tmp_path/'train';fixture(path,'router_train',range(4));audit_capture(path,'router_train')
    with pytest.raises(ValueError):audit_capture(path,'router_dev')
    record=json.loads((path/'manifest.json').read_text());record['samples'][0]['init_state_index']=5;write(path/'manifest.json',record)
    with pytest.raises(ValueError):audit_capture(path,'router_train')
    for role,offset,count in [('router_train',4,1),('router_dev',3,1),('development_evaluation',4,46)]:
        with pytest.raises(ValueError):check_range(role,offset,count)
def test_selector_cli_grouped_train_dev_save_and_hash_rejection(tmp_path):
    train,dev=tmp_path/'train',tmp_path/'dev';tr=fixture(train,'router_train',range(4));dr=fixture(dev,'router_dev',[4])
    tl,dl=tmp_path/'tl',tmp_path/'dl';labels(tl,train,tr);labels(dl,dev,dr)
    schema=tmp_path/'schema.json';write(schema,CausalFeatureEncoder([f'task {t}' for t in range(10)]).schema())
    stats=tmp_path/'stats.json';write(stats,{'q01':[-1.]*8,'q99':[1.]*8,'mask':[True]*8})
    output=tmp_path/'router';command=[sys.executable,str(ROOT/'scripts/train_b5_router.py'),'--train',str(train),'--dev',str(dev),'--train-labels',str(tl),'--dev-labels',str(dl),'--norm-stats',str(stats),'--feature-schema',str(schema),'--output',str(output)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    artifact=json.loads((output/'artifact.json').read_text())
    assert artifact['parameters']==25922 and artifact['steps']==500
    assert artifact['adapter_sha256']==sha(output/'router.pt') and artifact['dev_accuracy']>.9
    receipt=json.loads((dl/'receipt.json').read_text());receipt['capture_sha256']='0'*64;write(dl/'receipt.json',receipt)
    command[-1]=str(tmp_path/'rejected');result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/'rejected').exists()
    # Different action/file bytes cannot conceal an identical policy input.
    target=json.loads((dev/'manifest.json').read_text());source=tr[0]
    with np.load(train/source['file']) as s:sample={k:s[k] for k in s.files}
    sample['action']=np.ones(7);name=target['samples'][0]['file'];np.savez(dev/name,**sample)
    target['samples'][0]['sha256']=sha(dev/name);target['samples'][0]['observation_sha256']=source['observation_sha256'];write(dev/'manifest.json',target)
    receipt['capture_sha256']=sha(dev/'manifest.json');write(dl/'receipt.json',receipt)
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and 'policy-observation content overlap' in result.stderr
