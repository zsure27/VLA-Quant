# 实验报告索引

当前总结：[2026-10-17 近三轮实验目的、结果分析与短期计划](experiments/p2-shared-peft/20261017-three-rounds-summary-and-short-plan/REPORT_CN.md)。B1时间覆盖以正式各250回合收束，后续执行[多类小规模探索策略](../docs/EXPLORATORY_QUANT_FINETUNING_20261017_CN.md)，不再恢复旧450回合续片。

报告按研究计划模块归档，原始小结果使用相同的模块名和会话名保存在 [`results/experiments/`](../results/experiments/README_CN.md)。当前重点是量化微调方法优化；[旧主线协议](../docs/PEFT_CONTEXTUAL_ROUTING_EXECUTION_20260925_CN.md)保留历史背景，执行时以最新探索策略和当前实验卡为准。每轮仍保存目的、实际条件、结果、成本、不确定性及下一步决定。

## 模块

| 模块 | 状态 | 内容 |
|---|---|---|
| [`p0-foundation-baselines`](experiments/p0-foundation-baselines/README_CN.md) | 已完成 | BF16/W4 基线、W2 视觉与语言归因、实现与数值诊断 |
| [`p1-data-contract`](experiments/p1-data-contract/README_CN.md) | 已通过 | 训练轨迹清单、按轨迹切分、泄漏审计、AWQ q/z/Δ 等价契约 |
| [`p2-shared-peft`](experiments/p2-shared-peft/README_CN.md) | 当前研究入口 | 初始化、仅初始化参照、时间覆盖与量化微调探索；含早期PEFT历史 |
| [`p3-static-backbone`](experiments/p3-static-backbone/README_CN.md) | 已冻结 | 视觉 group、W4 岛、层数裁剪与 16L/14L/12L Spatial500 |
| [`p4-expert-complementarity`](experiments/p4-expert-complementarity/README_CN.md) | 历史方向，当前不扩展 | 专家互补的设计和约束 |
| [`p5-contextual-routing`](experiments/p5-contextual-routing/README_CN.md) | 已有B5诊断，当前不扩展 | Top-1 Router、最佳固定和task-only对照 |
| [`secondary-smoothquant`](experiments/secondary-smoothquant/README_CN.md) | 次线 | SQ 数值契约与 W/A 分离，不与 AWQ 主结论混合 |

## 汇总与历史

- [2026-10-17 三轮总结](experiments/p2-shared-peft/20261017-three-rounds-summary-and-short-plan/REPORT_CN.md)：初始化、仅初始化和取帧时间的主要结果与短期计划。
- [`summaries/20260929-baseline-progress/`](summaries/20260929-baseline-progress/README_CN.md)：截至9月29日的AWQ、混精底座、PEFT、A1独立性审计；用于复核当时结论。
- [`summaries/20260918-experiment-overview/`](summaries/20260918-experiment-overview/)：综合结论、按实验名称的日志和证据索引。
- [`archive/20260916-weekend-review/`](archive/20260916-weekend-review/)：早期完整分析归档，仅保留历史语境。
- [`REPOSITORY_LAYOUT_MIGRATION_20260925.json`](../docs/REPOSITORY_LAYOUT_MIGRATION_20260925.json)：旧路径到新路径的机器可读映射。

## 会话命名与内容

统一路径为：

```text
reports/experiments/<module>/YYYYMMDD-<host-or-scope>-<experiment>/
results/experiments/<module>/YYYYMMDD-<host-or-scope>-<experiment>/
backups/experiments/<module>/YYYYMMDD-<host-or-scope>-<experiment>/
```

同一实测会话的报告目录与结果目录使用相同名称。报告保存分析、图和证据入口；结果保存实际命令、console、配置、逐回合指标、源码版本与哈希。跨轮综合报告只汇总其引用的原始结果，不代表新增实验。完整大归档优先使用GitHub Release附件；本机仅保留工作所需的按需下载和暂存，见[备份策略](../docs/GITHUB_BACKUP_POLICY_20261017_CN.md)与[云端索引](../backups/github/README_CN.md)。

只有预注册方案、尚无实测结果时，将卡片放入 [`docs/pre-run-cards/`](../docs/pre-run-cards/README_CN.md)，不占用实验报告会话目录；真实执行后再建立同名报告与结果目录。

历史 `backup/SHA256SUMS.txt` 和关机回执继续记录当时的服务器路径，不为新目录布局改写。每轮新实验必须在启动前选定模块；不确定归类时先放入当前门禁对应模块，不能重新创建 `reports/sessions/`、顶层 `results/<session>/` 或 `results/pending-*`。
