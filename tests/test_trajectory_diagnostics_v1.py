import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from qvla.trajectory_diagnostics_v1 import rotation_difference, sustained_windows, same_observation_chunk, load_steps, compare_episode
from qvla.trajectory_capture_v1 import observe_episode, CaptureFailure


class GeometryTests(unittest.TestCase):
    def test_quaternion_sign_and_signed_angle(self):
        angle=np.pi/2
        q=np.array([0,0,np.sin(angle/2),np.cos(angle/2)])
        np.testing.assert_allclose(rotation_difference(q,-q),[0,0,0],atol=1e-10)
        np.testing.assert_allclose(rotation_difference(q,[0,0,0,1]),[0,0,angle],atol=1e-10)
        np.testing.assert_allclose(rotation_difference([0,0,0,1],q),[0,0,-angle],atol=1e-10)

    def test_persistence_null_and_single_spike(self):
        self.assertEqual(sustained_windows([0]*20,1,3),[])
        self.assertEqual(sustained_windows([0,2,0],1,3),[])
        self.assertEqual(sustained_windows([0,2,3,2,0],1,3)[0]["confirmed_index"],3)

    def test_wrong_input_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): same_observation_chunk(np.zeros((8,7)),np.zeros((8,7)),"a","b")
        a=np.zeros((8,7));a[0,0]=np.nan
        with self.assertRaises(ValueError): same_observation_chunk(a,np.zeros((8,7)),"a","a")

    def test_duplicate_steps_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"steps.jsonl"
            p.write_text('\n'.join(json.dumps({"kind":"control_step","control_step":0}) for _ in range(2)))
            with self.assertRaises(ValueError):load_steps(p)

    def test_pair_mismatch_and_censoring(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/"a",Path(td)/"b"
            a.mkdir();b.mkdir()
            meta=dict(suite="libero_goal",task_id=0,reset_index=5,model_seed=7,env_seed=0,
                init_state_sha256="s",checkpoint_sha256="c",cohort_id="test",chunk_size=8,
                model_id="candidate",status="timeout",data_kind="synthetic_demo")
            row=dict(kind="control_step",control_step=0,post_position_m=[0,0,0],post_quaternion_xyzw=[0,0,0,1])
            for folder in (a,b):
                (folder/"episode.json").write_text(json.dumps(meta))
                (folder/"steps.jsonl").write_text(json.dumps(row))
            row["control_step"]=1
            with (a/"steps.jsonl").open("a") as f:f.write("\n"+json.dumps(row))
            thresholds=dict(translation_m=.02,rotation_deg=5,persistence=3)
            result=compare_episode(a,b,thresholds)
            self.assertTrue(result["censored"])
            self.assertEqual(result["common_steps"],1)
            meta["env_seed"]=99
            (b/"episode.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):compare_episode(a,b,thresholds)


class ObserverTests(unittest.TestCase):
    def run_mock(self, root, early=False):
        obs={"robot0_eef_pos":np.zeros(3),"robot0_eef_quat":np.array([0,0,0,1]),"robot0_gripper_qpos":np.zeros(2)}
        class Env:
            calls=0
            def step(self,a):
                self.calls+=1
                return {k:v.copy() for k,v in obs.items()},0,self.calls==3,{}
            def set_init_state(self,s):return {k:v.copy() for k,v in obs.items()}
            def get_observation(self):return {k:v.copy() for k,v in obs.items()}
        env=Env()
        cfg=SimpleNamespace(seed_protocol="paired",initial_states_path="DEFAULT",task_suite_name="libero_spatial",num_open_loop_steps=2,num_steps_wait=1)
        chunk=np.arange(14,dtype=float).reshape(2,7)
        L=SimpleNamespace(get_action=lambda *a,**k:chunk.copy(),log_message=lambda *a,**k:None,TASK_MAX_STEPS={"libero_spatial":2})
        received=[]
        def episode(cfg,env,task_description,model=None):
            env.set_init_state(None);env.step([0]*7)
            if early:return False,[]
            o={"full_image":np.zeros((2,2,3),dtype=np.uint8),"wrist_image":np.zeros((2,2,3),dtype=np.uint8),"state":np.zeros(8)}
            a=L.get_action(cfg,None,o,"task")
            for x in a:received.append(x.copy());env.step(x)
            return True,[]
        L.run_episode=episode
        old_action=L.get_action
        old_step=env.step
        with observe_episode(L,root,{"data_kind":"synthetic_demo","model_id":"mock","checkpoint_sha256":"0"*64,"cohort_id":"mock"}):
            L.log_message('EPISODE_MANIFEST '+json.dumps(dict(task_id=0,init_state_index=5,model_seed=7,env_seed=0,init_state_sha256="s")))
            L.run_episode(cfg,env,"task")
        self.assertIs(L.run_episode,episode)
        self.assertIs(L.get_action,old_action)
        self.assertEqual(env.step,old_step)
        np.testing.assert_array_equal(received,chunk)
        self.assertEqual(env.calls,3)
        return root/"task-00-reset-005"

    def test_observer_preserves_action_count_and_records_every_step(self):
        with tempfile.TemporaryDirectory() as td:
            folder=self.run_mock(Path(td)/"capture")
            rows=load_steps(folder/"steps.jsonl")
            self.assertEqual([r["chunk_position"] for r in rows],[0,1])
            self.assertEqual(json.loads((folder/"episode.json").read_text())["status"],"success")

    def test_swallowed_early_error_is_not_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(CaptureFailure):self.run_mock(Path(td)/"capture",early=True)
            meta=json.loads((Path(td)/"capture/task-00-reset-005/episode.json").read_text())
            self.assertEqual(meta["status"],"interrupted_or_error")


if __name__=="__main__":unittest.main()
