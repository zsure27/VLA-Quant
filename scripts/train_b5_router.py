"""Fixed-budget CPU router; reset0-3 train, reset4 diagnostic, no eval labels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import VERSION,audit_capture,sha,save,source_hashes
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','dev','train-labels','dev-labels','norm-stats','feature-schema','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task-only',action='store_true');a=p.parse_args()
    import numpy as np,torch
    from qvla.b5_features import CausalFeatureEncoder,RAW_STATE
    from qvla.b5_runtime import Router
    torch.set_num_threads(1);torch.manual_seed(7)
    encoder=CausalFeatureEncoder.from_schema(json.loads(a.feature_schema.read_text()));norm=json.loads(a.norm_stats.read_text())
    def data(directory,labels,role):
        manifest=audit_capture(directory,role);receipt=json.loads((labels/'receipt.json').read_text())
        if receipt.get('gate')!='PASS_B5_SAME_OBSERVATION_REGRET' or receipt.get('role')!=role or receipt['capture_sha256']!=sha(directory/'manifest.json') or receipt['source_sha256']!=source_hashes() or sha(labels/'actions.pt')!=receipt['actions_sha256']:raise ValueError('Invalid label/data provenance')
        errors=torch.load(labels/'actions.pt',map_location='cpu',weights_only=True);features=[];targets=[];weights=[]
        if set(errors)!={r['file'] for r in manifest['samples']}:raise ValueError('Label coverage mismatch')
        for r in manifest['samples']:
            with np.load(directory/r['file'],allow_pickle=False) as s:
                features.append(encoder.encode({'full_image':s['router_image'],'wrist_image':s['router_wrist_image'],'state':s['state']},str(s['instruction'].item()),state_space=RAW_STATE,norm_stats=norm))
            e=errors[r['file']]
            if e.shape!=(2,) or not torch.isfinite(e).all() or (e<0).any():raise ValueError('Invalid teacher regret')
            targets.append(0 if e[0]<e[1] else 1);weights.append(float(abs(e[0]-e[1]))+1e-5)
        return torch.tensor(np.stack(features)),torch.tensor(targets),torch.tensor(weights),manifest
    x,y,w,train=data(a.train,a.train_labels,'router_train');dx,dy,dw,dev=data(a.dev,a.dev_labels,'router_dev')
    if {r['observation_sha256'] for r in train['samples']}&{r['observation_sha256'] for r in dev['samples']}:raise ValueError('Train/dev policy-observation content overlap')
    if a.output.exists():raise ValueError('Never overwrite router artifact')
    model=Router(a.task_only);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    for step in range(500):
        ids=torch.randperm(len(x))[:64];opt.zero_grad();loss=(torch.nn.functional.cross_entropy(model(x[ids]),y[ids],reduction='none')*w[ids]).sum()/w[ids].sum();loss.backward();opt.step()
    model.eval();a.output.mkdir(parents=True);torch.save(model.state_dict(),a.output/'router.pt')
    save(a.output/'artifact.json',{'version':VERSION,'gate':'PASS_B5_ROUTER_TRAIN_CONTRACT','steps':500,'task_only':a.task_only,
        'architecture':[10 if a.task_only else 402,64,2],'parameters':sum(p.numel() for p in model.parameters()),'adapter_sha256':sha(a.output/'router.pt'),
        'feature_schema':encoder.schema(),'feature_sha256':encoder.schema_sha256,'norm_stats_sha256':sha(a.norm_stats),'source_sha256':source_hashes(),
        'train_capture_sha256':sha(a.train/'manifest.json'),'dev_capture_sha256':sha(a.dev/'manifest.json'),
        'train_label_sha256':sha(a.train_labels/'receipt.json'),'dev_label_sha256':sha(a.dev_labels/'receipt.json'),
        'train_accuracy':float((model(x).argmax(1)==y).float().mean()),'dev_accuracy':float((model(dx).argmax(1)==dy).float().mean()),
        'claim':'CPU selector fit only; no closed-loop benefit; no dev checkpoint selection', 'tie_rule':'E1'})
if __name__=='__main__':main()
