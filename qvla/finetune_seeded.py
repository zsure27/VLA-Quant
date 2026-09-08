"""固定版本 OFT 的单 GPU 训练入口：先设置随机源，再导入训练/构建 RLDS。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-seed", type=int, default=17)
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("此快速微调入口只支持单 GPU；多卡 RLDS 分片与跨 rank 种子尚未验证")
    if any(x.startswith("--resume") for x in remaining):
        raise RuntimeError("上游不保存完整 RNG/数据游标，不能保证精确续训；请从新运行开始")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, __file__] + sys.argv[1:])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from qvla.runtime_contract import assert_oft_runtime
    assert_oft_runtime()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # TF 只处理数据，不抢占模型显存
    from qvla.reproducibility import seed_all
    seed_all(args.seed, strict=True, tensorflow=True)
    from prismatic.vla.datasets import RLDSDataset
    original = RLDSDataset.make_dataset

    def seeded_dataset(self, config):
        # 数据管线独立于模型初始化消耗的随机数；验证集与训练集分离。
        from qvla.reproducibility import derive_seed
        stream_seed = derive_seed(args.data_seed, "train" if config["train"] else "validation")
        tf.random.set_seed(stream_seed)
        result = original(self, config)
        options = tf.data.Options()
        options.experimental_deterministic = True
        return (result[0].with_options(options),) + tuple(result[1:])

    RLDSDataset.make_dataset = seeded_dataset
    target = Path(__file__).resolve().parents[1] / "vla-scripts/finetune.py"
    print(f"SEED_MANIFEST model={args.seed} data={args.data_seed} strict=True single_gpu=True")
    sys.argv = [str(target)] + remaining
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        RLDSDataset.make_dataset = original


if __name__ == "__main__":
    main()
