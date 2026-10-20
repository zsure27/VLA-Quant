# 本轮归档清单

服务器持久盘根目录：`/root/autodl-tmp/qvla-repro/backups/transfer/`。本机根目录：`C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/backups/experiments/p2-shared-peft/`。Git 仓库保留同名 `reports/experiments/p2-shared-peft/` 和 `results/experiments/p2-shared-peft/` 中的报告、小型原始指标、来源清单与分析代码；大模型输出和图像仅放双份完整归档。

| 归档 | 文件 | SHA256 |
|---|---|---|
| 前轮 P2.5 及 A/B/C 训练 | `20260927-107-p25-and-matrix/20260927-p25-and-matrix.tar.gz` | `48ab0b163c7edd4379108fc2727ab6eab9657e963b422fed729e254d8f5fd492` |
| router_dev 295 帧和 A/B/C/D | `20260927-107-router-dev/full.tar` | `03b864d615c7d31c828e8373713c4235c4bea8e6d268b83b9ff44672aedd541c` |
| P2.5 学生状态无 LoRA 控制 | `20260927-107-p25-student-state-control/full.tar` | `f9e353d3265b130538345f0c40bc6f32d0e5be1b761de55f222a3798543287dd` |

服务器对应文件名分别为 `20260927-p25-and-matrix.tar.gz`、`20260927-router-dev-full.tar`、`20260927-p25-state-control-full.tar`。报告的 SHA 指整个 tar 文件。前三个服务器原始 session 目录仍在持久盘，不依赖临时系统盘；归档不删除原件。GPU 评测器和训练脚本版本可由各 session 的 manifest、计划与提交记录追溯；B 训练后的包装器退出 127 曾由环境中裸 `python` 缺失导致，权重和步数已完成，原错误日志保留，C 修复为绝对 Python 路径。
