# P2｜量化补偿与恢复微调

## 当前入口 · 2026-10-17

先读[近三轮实验总结与短期计划](20261017-three-rounds-summary-and-short-plan/REPORT_CN.md)，再按[小规模探索策略](../../../docs/EXPLORATORY_QUANT_FINETUNING_20261017_CN.md)选择具体问题。当前不启动或恢复已结束计划，不默认每个候选跑满450回合。

| 对照 | 正式结果 | 主要证据 |
|---|---|---|
| 相同训练预算，普通零输出初始化Z／Response-SVD初始化S | Z 2/450；S 426/450 | [046初始化实验](20261015-046-quant-finetuning/) |
| PTQ／SVD仅初始化S0，不做梯度训练 | PTQ 77/450；S0 184/450 | [M0完整分析](20261016-091-residual-resume-v2/M0_450_ANALYSIS_CN.md) |
| 同40轨迹各80观测，Early80／Phase80取帧 | Early 241/250；Phase 110/250 | [B1累计250分析](20261016-091-residual-resume-v2/B1_RESET_20_29_ANALYSIS_CN.md) |

三组使用各自冻结身份与协议，不跨运行拼接分母。B1时间覆盖已收束；中断片不计成绩，微测重叠不增加正式样本数。这些仍是同十任务开发证据，不能称未见任务泛化。

## 历史研究

- [9月29日基线报告](../../summaries/20260929-baseline-progress/README_CN.md)记录12L底座、旧Scale-PEFT、旧Recovery-LoRA与学生状态重标C3；历史数值不被后来的全W2结果覆盖。
- 历史[`awq-w2a16-12l-mixed-spatial-v1`](../../../configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)仍保留，用于恢复相应实验，不能当成所有新模型的默认底座。
- `20260917-107-response-svd`为零步初始化测量；后来的训练模型和M0的S0身份应分别核对。
- 110的`env_seed=1`条件判为`A1_INVALID_CONDITION`，不计独立复现。真实输入、数值与数据角色约束见[评测门禁](../../../docs/EVALUATION_SMOKE_GATE_20260929_CN.md)。

原始小结果在同名[`results/experiments/p2-shared-peft/`](../../../results/experiments/p2-shared-peft/README_CN.md)，完整大归档见[GitHub资产索引](../../../backups/github/README_CN.md)。
