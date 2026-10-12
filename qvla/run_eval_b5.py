"""Versioned causal B5 evaluator/capture using the real LIBERO run_task chain."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import VERSION,sha,save,check_range,require_gpu_admission,require_bank_smoke,source_hashes
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('materials','feature-schema','norm-stats','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--role',choices=('router_train','router_dev','development_evaluation'),required=True)
    p.add_argument('--offset',type=int,required=True);p.add_argument('--count',type=int,required=True)
    p.add_argument('--policy',choices=('fixed-e0','fixed-e1','router','task-only','random'),required=True)
    p.add_argument('--router',type=Path);a=p.parse_args()
    check_range(a.role,a.offset,a.count)
    if a.role!='development_evaluation' and a.policy not in ('fixed-e0','fixed-e1'):raise ValueError('Only frozen experts collect training observations')
    m=json.loads(a.materials.read_text());require_gpu_admission(m);require_bank_smoke(m)
    for key in ('oft_root','libero_root'):sys.path.insert(0,m[key])
    import numpy as np,torch
    import experiments.robot.libero.run_libero_eval as L
    from libero.libero import benchmark
    from qvla.b5_features import CausalFeatureEncoder,RAW_STATE
    from qvla.b5_runtime import initialize,Router
    from qvla.action_jacobian_batch import prepare_inputs
    from qvla.on_policy_capture import observation_hash
    from experiments.robot.openvla_utils import prepare_images_for_vla
    encoder=CausalFeatureEncoder.from_schema(json.loads(a.feature_schema.read_text()));norm=json.loads(a.norm_stats.read_text())
    selector=None
    if a.policy in ('router','task-only'):
        if a.router is None:raise ValueError('Router artifact required')
        r=json.loads((a.router/'artifact.json').read_text())
        if r.get('version')!=VERSION or r.get('gate')!='PASS_B5_ROUTER_TRAIN_CONTRACT' or r.get('source_sha256')!=source_hashes() or r['feature_sha256']!=encoder.schema_sha256 or r['norm_stats_sha256']!=sha(a.norm_stats) or r['task_only']!=(a.policy=='task-only') or sha(a.router/'router.pt')!=r['adapter_sha256']:raise ValueError('Router feature/data/source identity mismatch')
        selector=Router(r['task_only']);selector.load_state_dict(torch.load(a.router/'router.pt',map_location='cpu',weights_only=True));selector.eval()
    if a.output.exists():raise ValueError('Never overwrite/repeat an evaluation or capture')
    a.output.mkdir(parents=True);save(a.output/'invocation.json',{'argv':sys.argv,'source_sha256':source_hashes(),'materials_sha256':sha(a.materials),'role':a.role,'offset':a.offset,'count':a.count,'policy':a.policy})
    cfg,model,head,proprio,processor,bank,route,handles=initialize(m)
    cfg.seed_protocol='paired';cfg.env_seed=1;cfg.seed=0;cfg.initial_state_offset=a.offset;cfg.num_trials_per_task=a.count
    cfg.num_open_loop_steps=8;cfg.num_steps_wait=10;cfg.initial_states_path='DEFAULT'
    L.validate_config(cfg);suite=benchmark.get_benchmark_dict()['libero_spatial']()
    instructions=[suite.get_task(t).language for t in range(10)]
    if tuple(' '.join(x.lower().split()) for x in instructions)!=encoder.instruction_vocabulary:raise ValueError('Task/instruction mapping differs')
    expected_norm=model.norm_stats[cfg.unnorm_key]['proprio']
    if json.dumps(expected_norm,sort_keys=True)!=json.dumps(norm,sort_keys=True):raise ValueError('Router and actual policy normalization differ')
    frozen_versions={n:p._version for n,p in model.named_parameters()};before=bank.frozen_sha256()
    if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for p in bank.parameters()):raise ValueError('Expert/base not frozen')
    current={'manifest':None,'captures':[],'query':0};samples=[];episodes=[]
    original={name:getattr(L,name) for name in ('get_action','get_libero_env','run_episode','log_message','save_rollout_video')}
    stream=(a.output/'trace.jsonl').open('x');rng=np.random.default_rng(7)
    def record(row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    class EnvProxy:
        def __init__(self,env):self.env=env;self.steps=0
        def __getattr__(self,name):return getattr(self.env,name)
        def reset(self,*args,**kwargs):self.steps=0;return self.env.reset(*args,**kwargs)
        def step(self,action):
            if self.steps>=cfg.num_steps_wait and not route.remaining_actions:raise ValueError('Environment step outside an admitted action chunk')
            result=self.env.step(action)
            if route.remaining_actions:route.record_executed_step()
            self.steps+=1
            return result
    def env(*args,**kwargs):
        value,instruction=original['get_libero_env'](*args,**kwargs);return EnvProxy(value),instruction
    def log(message,log_file=None):
        if str(message).startswith('EPISODE_MANIFEST '):current['manifest']=json.loads(str(message).split(' ',1)[1])
        if str(message).startswith('Episode error:'):record({'event':'episode_error','error':str(message)})
        original['log_message'](message,log_file)
    def episode(*args,**kwargs):
        r=current['manifest'];check_range(a.role,r['init_state_index'],1)
        key=f"{r['task_id']}:{r['init_state_index']}";route.begin_episode(key)
        current.update(captures=[],query=0);record({'event':'episode_begin','manifest':r})
        try:
            result=original['run_episode'](*args,**kwargs)
            if a.role!='development_evaluation':
                captured=current['captures']
                if not captured:raise ValueError('Empty capture episode')
                indices=np.unique(np.rint(np.linspace(0,len(captured)-1,min(5,len(captured)))).astype(int))
                for q in indices:
                    sample=captured[q];name=f"expert-{a.policy[-1]}-task-{r['task_id']:02d}-reset-{r['init_state_index']:02d}-query-{q:03d}.npz"
                    np.savez_compressed(a.output/name,**sample)
                    from qvla.b5_features import normalized_proprio
                    samples.append({'file':name,'sha256':sha(a.output/name),'observation_sha256':observation_hash(sample['router_image'],sample['router_wrist_image'],normalized_proprio(sample['state'],state_space=RAW_STATE,norm_stats=norm),sample['instruction']),'processed_inputs':json.loads(str(sample['processed_inputs_json'])),'collector_expert':int(a.policy[-1]),'task_id':r['task_id'],'init_state_index':r['init_state_index'],'query_in_episode':int(q),'init_state_sha256':r['init_state_sha256']})
            ending=route.end_episode();ending.update(success=bool(result[0]),manifest=r);episodes.append(ending);record({'event':'episode_end',**ending});return result
        except BaseException:
            if route.active_token is not None:route.abort_query(route.active_token)
            if route.episode_key is not None:route.end_episode()
            raise
    def action(cfg,model,obs,instruction,**kwargs):
        observation={k:np.asarray(obs[k]).copy() for k in ('full_image','wrist_image','state')}
        policy_images=prepare_images_for_vla([observation['full_image'],observation['wrist_image']],cfg)
        feature_observation={'full_image':np.asarray(policy_images[0]),'wrist_image':np.asarray(policy_images[1]),'state':observation['state']}
        features=encoder.encode(feature_observation,instruction,state_space=RAW_STATE,norm_stats=norm)
        expert=(int(a.policy[-1]) if a.policy.startswith('fixed-') else int(rng.integers(0,2)) if a.policy=='random' else int(selector(torch.from_numpy(features)).argmax()))
        sample={'image':observation['full_image'],'wrist_image':observation['wrist_image'],'state':observation['state'],'instruction':instruction}
        inputs,state=prepare_inputs(sample,cfg,model,processor,torch.device('cuda:0'))
        input_hash={k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in inputs.items()}
        input_hash['normalized_proprio']=hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
        token=route.begin_query(route.episode_key,current['query'],expert)
        try:
            with torch.no_grad():actions=original['get_action'](cfg,model,obs,instruction,**kwargs)
            chunk=np.asarray(actions)
            if chunk.shape!=(8,7) or not np.isfinite(chunk).all():raise ValueError('Wrong/nonfinite B5 action chunk')
            route.end_query(token);record({'event':'query','episode':token.episode_key,'query':current['query'],'expert':expert,'features_sha256':hashlib.sha256(features.tobytes()).hexdigest(),'processed_inputs':input_hash,'observation_sha256':observation_hash(observation['full_image'],observation['wrist_image'],np.asarray(state,dtype=np.float32),instruction),'raw_state':observation['state'].tolist(),'raw_policy_chunk':chunk.tolist()})
            if a.role!='development_evaluation':
                sample['action']=chunk[0].copy()  # student action; unused by teacher loss, never GT
                sample['router_image']=feature_observation['full_image'].copy()
                sample['router_wrist_image']=feature_observation['wrist_image'].copy()
                sample['processed_inputs_json']=np.array(json.dumps(input_hash,sort_keys=True))
                current['captures'].append(sample)
            current['query']+=1;return actions
        except BaseException:
            if route.active_token is not None:route.abort_query(token)
            raise
    try:
        L.get_action=action;L.get_libero_env=env;L.run_episode=episode;L.log_message=log;L.save_rollout_video=lambda *x,**k:None
        total=successes=0
        for task in range(10):total,successes=L.run_task(cfg,suite,task,model,L.get_image_resize_size(cfg),processor,head,proprio,total_episodes=total,total_successes=successes)
        if len(episodes)!=10*a.count or any(e['aborted'] for e in episodes):raise ValueError('Incomplete/aborted B5 episodes')
        if before!=bank.frozen_sha256() or frozen_versions!={n:p._version for n,p in model.named_parameters()}:raise ValueError('Frozen base/expert mutated')
        save(a.output/'manifest.json',{'version':VERSION,'role':a.role,'state_space':RAW_STATE,'proprio_norm_stats':norm,'samples':samples,'episodes':episodes,'policy':a.policy,'source_sha256':source_hashes(),'bank_parameters':bank.parameter_count,'bank_tensor_bytes':bank.tensor_bytes,'bank_sha256':before,'feature_schema_sha256':encoder.schema_sha256,'successes':successes,'trace_sha256':sha(a.output/'trace.jsonl'),'claim':'same-task development; no independent holdout claim'})
    finally:
        stream.close()
        for name,value in original.items():setattr(L,name,value)
        bank.close()
        for h in handles:h.remove()
if __name__=='__main__':main()
