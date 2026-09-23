# 量化与评估代码

这里是当前维护的 OpenVLA-OFT 适配代码；`legacy/qvla/` 保存早期 W2 实现，不默认安装。`scripts/install_adapter.sh` 把本目录的 `.py` 文件安装到固定的 OFT checkout，历史运行仍应以各分片记录的代码 SHA 和快照复核。

| 职责 | 主要文件 |
|---|---|
| 目标与契约 | `baseline_contract.py`、`build_connected_targets.py`、`causal_target_inventory.py`、`forward_target_inventory.py`、`runtime_contract.py`、`reproducibility.py` |
| 校准与伪量化 | `extract_balanced_calibration.py`、`calibrate_official_quant.py`、`official_quant_adapter.py`、`awq_block.py` |
| 评估入口 | `run_eval_official_quant.py`、`run_eval_component_w.py`、`run_eval_component_hook.py` |
| 动作敏感度与候选分配 | `action_jacobian_batch.py`、`aggregate_action_proxy.py`、`assign_gates_from_sensitivity.py`、`assign_gates_marginal.py` |
| 辅助训练与对照 | `finetune_seeded.py`、`compare_single_batch.py` |

当前评估为 BF16 权重承载的 fake quant。数值结果、闭环成功率和真实 packed 低比特执行分别验收；不能用这里的模型运行时间推断 INT2/INT4 加速。优先通过 `scripts/` 中的有限分片入口运行，核对 checkpoint/profile/源码哈希和任务初态后再扩大评测。
