"""区分模型、环境、数据随机流；不把设置种子等同于跨硬件逐位确定性。"""
from __future__ import annotations

import hashlib
import os
import random


def derive_seed(seed: int, *labels: object) -> int:
    if not 0 <= seed < 2**32:
        raise ValueError("种子必须在 [0, 2**32) 内")
    value = "/".join(map(str, (seed,) + labels)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def seed_all(seed: int, strict: bool = False, tensorflow: bool = False):
    import numpy as np
    import torch
    derive_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if strict:
        # 必须在首次 CUDA 矩阵运算前调用；不支持的算子直接报错。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if strict:
            tf.config.experimental.enable_op_determinism()
    return {"seed": seed, "strict": strict, "tensorflow_seeded": tensorflow,
            "pythonhashseed_at_launch": os.environ.get("PYTHONHASHSEED"),
            "note": "PYTHONHASHSEED 必须在解释器启动前设置；运行中赋值不会改变当前哈希盐"}


def episode_seeds(model_seed, env_seed, suite, task_id, episode_idx):
    # 不依赖上一回合运行多久，所有候选与教师按同一个键配对。
    return (derive_seed(model_seed, "model", suite, task_id, episode_idx),
            derive_seed(env_seed, "env", suite, task_id, episode_idx))
