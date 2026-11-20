"""Full ten-step action-graph prerequisite; no QVLA scores or formal episodes.

Unwrap only the native no_grad decorator. Saved activation offload changes
storage, not inference math. A normalized-action VJP is a connectivity probe,
not the physical-action sensitivity required by the complete QVLA method.
"""
import argparse
import gc
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from qvla.infra.common import digest, now, write_json
from qvla.pi05_torch_backend_v2 import BACKEND_ID, load_policy, target_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--audit-suite')
    args = p.parse_args()
    out = args.stage / 'action-graph'
    if args.audit_suite:
        result = json.loads((out / 'result.json').read_text())
        assert result['gate'] == 'PASS_PI05_NATIVE_TEN_STEP_ACTION_GRAPH_PREREQUISITE'
        assert args.audit_suite in [r['suite'] for r in result['checks']]
        print(json.dumps(dict(gate=result['gate'], suite=args.audit_suite, formal_episodes=0)))
        return
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    capture = args.root / 'rounds/20261111-084-pi05-smoothquant-a8-v2/capture'
    assert digest(capture / 'statistics.pt') == json.loads((capture / 'result.json').read_text())['profile_sha256']
    cache = torch.load(capture / 'statistics.pt', map_location='cpu', weights_only=True)
    rows = [next(r for r in cache['samples'] if r['suite'] == s) for s in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']]
    assert all(r['role'] == 'calibration' and r['reset_index'] == 0 for r in rows)
    policy = load_policy(args.root)
    model = policy._model
    from openpi.models.model import Observation
    import jax
    names = target_names(model)
    chosen = [next(n for n in names if 'vision_tower' in n and n.endswith('mlp.fc2')),
              next(n for n in names if 'paligemma.model.language_model' in n and n.endswith('q_proj')),
              next(n for n in names if 'gemma_expert' in n and n.endswith('q_proj'))]
    # Frozen parameter storage is already resident; offload only intermediates.
    protected = {v.untyped_storage().data_ptr() for v in model.parameters()}
    offloaded = 0

    def pack(tensor):
        nonlocal offloaded
        if tensor.device.type == 'cuda' and tensor.untyped_storage().data_ptr() not in protected:
            offloaded += tensor.numel() * tensor.element_size()
            if offloaded > 32 * 1024**3:
                raise RuntimeError('Action graph exceeds declared 32GiB saved-activation CPU budget')
            return tensor.device, tensor.detach().cpu()
        return None, tensor.detach()

    def unpack(saved):
        device, tensor = saved
        return tensor if device is None else tensor.to(device)

    checks = []
    for row in rows:
        offloaded = 0
        assert digest(row['path']) == row['sha256']
        with np.load(row['path'], allow_pickle=False) as sample:
            obs = {'observation/image': sample['image'], 'observation/wrist_image': sample['wrist_image'],
                   'observation/state': sample['state'], 'prompt': str(sample['prompt'].item())}
            noise = sample['policy_noise'].copy()
            expected = sample['raw_policy_chunk'].copy()
        native = np.asarray(policy.infer(obs, noise=noise)['actions'])
        assert np.array_equal(native, expected), 'Inherited native reference differs'
        inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to('cuda:0')[None, ...], inputs)
        observation = Observation.from_dict(inputs)
        for image in observation.images.values():
            assert image.is_floating_point()
            image.requires_grad_(True)
        noise_tensor = torch.from_numpy(noise).to('cuda:0')[None, ...].requires_grad_(True)
        stats = {name: dict(calls=0, gradients=0, finite=True, squared_norm=0.0) for name in chosen}
        handles = []
        for name in chosen:
            def forward(module, args, output, name=name):
                assert output.requires_grad, name
                stats[name]['calls'] += 1
                def backward(grad):
                    stats[name]['gradients'] += 1
                    stats[name]['finite'] &= bool(torch.isfinite(grad).all())
                    stats[name]['squared_norm'] += float(grad.float().square().sum())
                output.register_hook(backward)
            handles.append(model.get_submodule(name).register_forward_hook(forward))
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack), torch.enable_grad():
                raw = inspect.unwrap(type(model).sample_actions)(model, 'cuda:0', observation, noise=noise_tensor, num_steps=10)
                actual = policy._output_transform({'state': inputs['state'][0].detach().cpu().numpy(), 'actions': raw[0].detach().cpu().numpy()})['actions']
                assert np.array_equal(actual, native), 'Differentiable forward changes native actions'
                generator = torch.Generator(device='cuda').manual_seed(7)
                sign = torch.randint(0, 2, (1, 10, 7), generator=generator, device='cuda') * 2 - 1
                objective = (raw[..., :7].float() * (sign / np.sqrt(70))).sum()
                valid_images = [image for name,image in observation.images.items() if bool(observation.image_masks[name].any())]
                assert valid_images
                gradients = torch.autograd.grad(objective, (noise_tensor, *valid_images))
                gradient = gradients[0]
                image_checks = [dict(finite=bool(torch.isfinite(g).all()), squared_norm=float(g.float().square().sum())) for g in gradients[1:]]
                write_json(out / ('vjp-' + row['suite'] + '.json'), dict(components=stats, image_gradients=image_checks, native_actions_bitwise=True))
                assert all(x['finite'] and x['squared_norm'] > 0 for x in image_checks)
            assert torch.isfinite(gradient).all() and float(gradient.square().sum()) > 0
            assert all(v['gradients'] == v['calls'] and v['finite'] and v['squared_norm'] > 0 for v in stats.values())
            assert all(not v.requires_grad and v.grad is None for v in model.parameters())
            checks.append(dict(suite=row['suite'], calibration_sha256=row['sha256'], native_actions_bitwise=True,
                               gradient_components=stats, noise_vjp_finite_nonzero=True))
            write_json(out / 'progress.json', dict(completed_suites=len(checks), formal_episodes=0))
        finally:
            for handle in handles:
                handle.remove()
        del raw, objective, gradient, gradients, valid_images, noise_tensor, observation, inputs
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(gate='PASS_PI05_NATIVE_TEN_STEP_ACTION_GRAPH_PREREQUISITE', backend=BACKEND_ID,
                  checks=checks, formal_episodes=0, qvla_full_admitted=False,
                  physical_action_score_and_bit_allocation_not_implemented=True,
                  native_source_sha256=digest(args.root / 'sources/openpi/src/openpi/models_pytorch/pi0_pytorch.py'),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), wall_seconds=time.monotonic()-started, finished_at=now())
    write_json(out / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
