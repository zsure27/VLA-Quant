"""Independent official OpenPI PyTorch cohort; never claims JAX equivalence."""
import dataclasses
from pathlib import Path
import re
import sys

import numpy as np

BACKEND_ID = 'openpi-pytorch-original-independent-v1'
SOURCE_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'


def target_names(model):
    import torch
    patterns = [
        r'paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))',
        r'paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))',
        r'paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))',
    ]
    return sorted(name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)
                  and any(re.fullmatch(p,name) for p in patterns))


def load_policy(root):
    import torch
    root=Path(root)
    sys.path[:0]=[str(root/'sources/openpi/src'),str(root/'sources/openpi/packages/openpi-client/src')]
    from openpi.training import config, checkpoints
    from openpi.policies import policy_config
    c=config.get_config('pi05_libero')
    c=dataclasses.replace(c,model=dataclasses.replace(c.model,pytorch_compile_mode=None))
    data=c.data.create(c.assets_dirs,c.model)
    stats=checkpoints.load_norm_stats(root/'models/pi05_libero/assets',data.asset_id)
    policy=policy_config.create_trained_policy(c,root/'models/pi05_libero_pytorch_bridge_v2',
                                              norm_stats=stats,pytorch_device='cuda:0')
    policy._model.requires_grad_(False)
    def reject_unused(*unused):
        raise RuntimeError('Unmapped vocabulary head entered the action path')
    policy._model.paligemma_with_expert.gemma_expert.lm_head.register_forward_pre_hook(reject_unused)
    assert all(not x.requires_grad for x in policy._model.parameters())
    assert not policy._model.training
    return policy


def noise_stream():
    # Query-index-paired noise is independent of model loading and internal RNG use.
    return np.random.default_rng(0)


def next_noise(stream):
    return stream.standard_normal((50,32)).astype(np.float32)
