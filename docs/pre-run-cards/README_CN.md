# 实验运行前卡片

本目录保存尚未产生配对结果的预注册方案。卡片不是实验成功率报告，也不单独创建 `reports/experiments/` 与 `results/experiments/` 会话目录。执行后，原始产物、分析和 gate 决定应进入同名的模块化会话目录，并在报告中回链对应卡片。

- [`A1_EVALUATION_REPLICATION_20260929_CN.md`](A1_EVALUATION_REPLICATION_20260929_CN.md)：冻结 C3 的评测条件复现预案，当前为历史 HOLD；`env_seed` 单独变化已被实测判定无效，不能直接按此卡派发 GPU 长实验。
- [`VISUAL_INCREMENTAL_LORA_20261005_CN.md`](VISUAL_INCREMENTAL_LORA_20261005_CN.md)：冻结 exact-12L 与 LW，探索新增视觉 rank8 adapter；旧 LW→视觉确认性门禁未过，目前 HOLD 代码和运行烟雾。
- [`PURE_W2_RECOVERY_LORA_20261005_CN.md`](PURE_W2_RECOVERY_LORA_20261005_CN.md)：独立的全合格量化层 W2 底座及单个语言 rank8 LoRA；目前 HOLD 版本化 profile、代码与运行烟雾。

两卡的阻断项和下次开机顺序见 [`PEFT_VISUAL_PURE_W2_CODE_REVIEW_20261005_CN.md`](../PEFT_VISUAL_PURE_W2_CODE_REVIEW_20261005_CN.md)；现有训练/测试样本量见 [`PEFT_DATA_SPLIT_AUDIT_20261005_CN.md`](../PEFT_DATA_SPLIT_AUDIT_20261005_CN.md)。

未来声称独立 seed/reset 的长闭环实验先执行 [`EVALUATION_SMOKE_GATE_20260929_CN.md`](../EVALUATION_SMOKE_GATE_20260929_CN.md)。
