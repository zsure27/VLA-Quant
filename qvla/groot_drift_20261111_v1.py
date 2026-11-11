"""GR00T native paired pilot binding. No training and no claimed low-bit kernel."""
import argparse
import copy
import gc
import hashlib
import json
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from qvla.execute_groot_bf16_episode_v2 import SUITES, verify_snapshot, sha
from qvla.infra.common import write_json, now


def seed(query):
    value = 7 + query
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    return value


def load(root, suite):
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    config = load_data_config('examples.Libero.custom_data_config:LiberoDataConfig')
    policy = Gr00tPolicy(model_path=str(root/'models'/SUITES[suite][0]),
                         embodiment_tag='new_embodiment', modality_config=config.modality_config(),
                         modality_transform=config.transform(), denoising_steps=8)
    policy.model.eval().requires_grad_(False)
    if any(not torch.isfinite(p).all() or p.requires_grad for p in policy.model.parameters()):
        raise ValueError('Nonfinite or unfrozen model')
    return policy


def infer(policy, obs, query):
    """Capture the actual native torch.randn noise, without changing the sampler."""
    value = seed(query)
    original = torch.randn
    noises = []
    def captured(*args, **kwargs):
        tensor = original(*args, **kwargs)
        noises.append(tensor.detach().float().cpu().numpy())
        return tensor
    torch.randn = captured
    try:
        raw = policy.get_action(copy.deepcopy(obs))
    finally:
        torch.randn = original
    from examples.Libero.eval.run_libero_eval import GR00TPolicy
    wrapper = object.__new__(GR00TPolicy)
    wrapper.action_keys = ['x','y','z','roll','pitch','yaw','gripper']
    action = wrapper._convert_to_libero_action(raw, 0)
    if action.shape != (7,) or not np.isfinite(action).all() or len(noises) != 1:
        raise ValueError('Native action/noise contract failed')
    expected = (1, policy.model.action_head.config.action_horizon, policy.model.action_head.config.action_dim)
    if noises[0].shape != expected:
        raise ValueError('Actual flow-noise shape differs from loaded model')
    return action, noises[0], value


def episode(policy, root, suite, output, reset, role, reference=None, admission=None):
    from examples.Libero.eval.run_libero_eval import GR00TPolicy, get_libero_env, get_libero_dummy_action
    from libero.libero import benchmark
    output.mkdir(parents=True, exist_ok=False)
    group = benchmark.get_benchmark_dict()[suite]()
    assert group.n_tasks == 10
    task = group.get_task(0)
    initial = np.asarray(group.get_task_init_states(0)[reset])
    report = dict(gate='RUNNING', suite=suite, task_id=0, reset_index=reset, role=role,
                  backend='groot-native-bf16-autocast-v1', model_seed=7, env_seed=7,
                  execute_first_actions=1, denoising_steps=8, queries=0, success=False,
                  initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(), argv=sys.argv)
    write_json(output/'identity.json', report)
    env = None
    start = time.perf_counter()
    frames = []
    try:
        env, _ = get_libero_env(task, resolution=256)
        env.seed(7)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done: raise ValueError('Early environment termination')
        wrapper = object.__new__(GR00TPolicy)
        wrapper.headless = True
        first_obs = None
        for step in range(SUITES[suite][3]):
            processed = wrapper._process_observation(obs, task.language)
            arrays = {k:np.asarray(v) for k,v in processed.items()}
            if any(a.dtype.kind != 'U' and not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('Nonfinite input')
            if step == 0 and reference:
                with np.load(reference/'query-00000.npz', allow_pickle=False) as ref:
                    if any(not np.array_equal(v,ref[k]) for k,v in arrays.items()):
                        raise ValueError('Reference first actual input differs')
            action, noise, rngseed = infer(policy, processed, step)
            if step == 0:
                first_obs = copy.deepcopy(processed)
                if admission:
                    admission(first_obs, action, noise)
            if reference and step == 0:
                with np.load(reference/'query-00000.npz', allow_pickle=False) as ref:
                    if not np.array_equal(noise,ref['flow_noise']):
                        raise ValueError('Reference first flow noise differs')
            before = {k:np.asarray(obs[k]).copy() for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}
            frames.append(np.asarray(processed['video.image'])[0])
            obs, reward, done, _ = env.step(action.tolist())
            after = {'post_'+k:np.asarray(obs[k]).copy() for k in before}
            np.savez_compressed(output/f'query-{step:05d}.npz', **arrays, **before, **after,
                                action=action, flow_noise=noise, query_seed=np.asarray(rngseed),
                                reward=np.asarray(reward), done=np.asarray(done))
            report['queries'] += 1
            if done:
                report['success'] = True
                break
        import imageio.v2 as imageio
        imageio.mimsave(output/'rollout.mp4', frames, fps=20, codec='libx264')
        report.update(gate='PASS_GROOT_PAIRED_TRACE', video_sha256=sha(output/'rollout.mp4'),
                      query_files={p.name:sha(p) for p in sorted(output.glob('query-*.npz'))})
    except BaseException as error:
        report.update(gate='FAIL_GROOT_PAIRED_TRACE',error=repr(error))
        raise
    finally:
        if env is not None: env.close()
        report['wall_seconds_including_admission'] = time.perf_counter()-start
        write_json(output/'result.json',report)
    return first_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--round-dir', type=Path, required=True)
    p.add_argument('--suite', choices=SUITES, required=True)
    args = p.parse_args()
    root, rd, suite = args.root, args.round_dir, args.suite
    assert socket.gethostname() == 'autodl-container-1421458302-7454c1c8'
    sys.path[:0] = [str(root/'sources/Isaac-GR00T'),str(root/'sources/LIBERO')]
    import mujoco
    assert mujoco.__version__ == '3.3.2'
    verify_snapshot(root/'models'/SUITES[suite][0],suite)
    target = rd/suite
    target.mkdir(exist_ok=True)
    policy = load(root,suite)
    def admission(obs, action, noise):
        repeated, n, _ = infer(policy,obs,0)
        assert np.array_equal(action,repeated) and np.array_equal(noise,n)
        write_json(target/'repeat-action-gate.json',dict(gate='PASS_GROOT_REPEAT_ACTION',
                   actual_flow_noise_shape=list(noise.shape),execute_first_actions=1,registered_at=now()))
    first = episode(policy,root,suite,target/'bf16',5,'development',admission=admission)
    a,n,_ = infer(policy,first,0)
    del policy
    gc.collect();torch.cuda.empty_cache()
    policy = load(root,suite)
    b,m,_ = infer(policy,first,0)
    assert np.array_equal(a,b) and np.array_equal(n,m)
    write_json(target/'reload-action-gate.json',dict(gate='PASS_GROOT_RELOAD_ACTION',checkpoint_unchanged=True))
    # Noise-floor control and calibration are explicitly excluded from success denominators.
    episode(policy,root,suite,target/'bf16-repeat',5,'repeat_control',reference=target/'bf16')
    episode(policy,root,suite,target/'calibration-reset0',0,'calibration')
    del policy
    gc.collect();torch.cuda.empty_cache()
    write_json(target/'reference-gate.json',dict(gate='PASS_GROOT_REFERENCE_REPEAT_AND_CALIBRATION',
               suite=suite,development_episodes=1,calibration_episodes=1,repeat_control_episodes=1))


if __name__ == '__main__': main()
