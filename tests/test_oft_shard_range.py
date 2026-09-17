import ast,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
class State:
 def __init__(self,i):self.i=i;self.dtype="test";self.shape=(1,)
 def tobytes(self):return str(self.i).encode()
class ShardRangeTest(unittest.TestCase):
 def call(self,offset,count):
  source=Path(__file__).resolve().parents[1]/"overlays/openvla-oft/experiments/robot/libero/run_libero_eval.py"
  tree=ast.parse(source.read_text(encoding="utf-8"));f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_task")
  messages=[];closed=[];observed=[]
  env=NS(close=lambda:closed.append(True),seed=lambda s:None)
  def episode(*args):observed.append(args[-2].i);return args[-2].i%2==0,[]
  ns=dict(GenerateConfig=object,tqdm=NS(tqdm=lambda x:x),np=NS(asarray=lambda x:x),json=json,
   load_initial_states=lambda *args:([State(i) for i in range(50)],None),get_libero_env=lambda *args,**kwargs:(env,"test"),
   log_message=lambda m,*args:messages.append(m),episode_seeds=lambda *args:(1000+args[-1],2000+args[-1]),seed_all=lambda *args,**kwargs:None,
   run_episode=episode,save_rollout_video=lambda *args,**kwargs:None)
  exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),"exec"),ns)
  cfg=NS(initial_state_offset=offset,num_trials_per_task=count,initial_states_path="DEFAULT",model_family="openvla",env_img_res=256,seed=0,env_seed=0,seed_protocol="paired",task_suite_name="libero_spatial",strict_determinism=False,use_wandb=False)
  result=ns["run_task"](cfg,NS(get_task=lambda i:object()),2,None,None)
  manifest=[json.loads(s.split(" ",1)[1]) for s in messages if s.startswith("EPISODE_MANIFEST ")]
  return result,observed,manifest,closed
 def test_shard_uses_real_indices_and_seeds(self):
  result,observed,manifest,closed=self.call(5,10)
  self.assertEqual(result,(10,5));self.assertEqual(observed,list(range(5,15)))
  self.assertEqual([m["init_state_index"] for m in manifest],observed)
  self.assertEqual([m["model_seed"] for m in manifest],list(range(1005,1015)));self.assertEqual(len(closed),1)
 def test_adjacent_shards_no_overlap(self):
  a=self.call(5,10)[1];b=self.call(15,10)[1];self.assertFalse(set(a)&set(b))
 def test_out_of_bounds_rejected(self):
  with self.assertRaises(ValueError):self.call(45,6)
if __name__=="__main__":unittest.main()
