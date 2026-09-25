# 实验报告索引

报告按研究计划模块归档，原始小结果使用相同的模块名和会话名保存在 [`results/experiments/`](../results/experiments/README_CN.md)。当前主线由 [`PEFT_CONTEXTUAL_ROUTING_EXECUTION_20260925_CN.md`](../docs/PEFT_CONTEXTUAL_ROUTING_EXECUTION_20260925_CN.md)约束。

## 模块

| 模块 | 状态 | 内容 |
|---|---|---|
| [`p0-foundation-baselines`](experiments/p0-foundation-baselines/README_CN.md) | 已完成 | BF16/W4 基线、W2 视觉与语言归因、实现与数值诊断 |
| [`p1-data-contract`](experiments/p1-data-contract/README_CN.md) | 下一阶段 | 训练轨迹清单、按轨迹切分、泄漏审计、AWQ q/z/Δ 等价契约 |
| [`p2-shared-peft`](experiments/p2-shared-peft/README_CN.md) | 初始化预实验已完成；训练待做 | shared Scale PEFT、Recovery LoRA、Response SVD 初始化 |
| [`p3-static-backbone`](experiments/p3-static-backbone/README_CN.md) | 已冻结 | 视觉 group、W4 岛、层数裁剪与 16L/14L/12L Spatial500 |
| [`p4-expert-complementarity`](experiments/p4-expert-complementarity/README_CN.md) | 有条件待做 | 两个等预算专家和完整 expert×context 矩阵 |
| [`p5-contextual-routing`](experiments/p5-contextual-routing/README_CN.md) | 有条件待做 | H17 因果上下文、Top-1 Router 与必要对照 |
| [`secondary-smoothquant`](experiments/secondary-smoothquant/README_CN.md) | 次线 | SQ 数值契约与 W/A 分离，不与 AWQ 主结论混合 |

## 汇总与历史

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

同一会话的报告目录与结果目录必须使用相同名称。报告目录保存 `README_CN.md`、图、分析用数据、manifest 和复现脚本；结果目录保存实际命令、console、配置、逐回合指标、源码版本与哈希。完整本地归档位于被 Git 忽略的 `backups/experiments/`，目录规则见 [`backups/README_CN.md`](../backups/README_CN.md)。

历史 `backup/SHA256SUMS.txt` 和关机回执继续记录当时的服务器路径，不为新目录布局改写。每轮新实验必须在启动前选定模块；不确定归类时先放入当前门禁对应模块，不能重新创建 `reports/sessions/`、顶层 `results/<session>/` 或 `results/pending-*`。
