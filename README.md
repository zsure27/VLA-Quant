# VLA 量化与微调实验

保存 OpenVLA-OFT 低比特量化的实现、配置、结果和分析。当前研究重点是通过多类、小规模、单变量实验，寻找有效的量化补偿与微调方法。

## 当前入口 · 2026-10-19

- **先读结论：**[近三轮实验总结与短期计划](reports/experiments/p2-shared-peft/20261017-three-rounds-summary-and-short-plan/REPORT_CN.md)。
- **后续怎么做：**[小规模探索策略](docs/EXPLORATORY_QUANT_FINETUNING_20261017_CN.md)。差异已清楚的候选及时收束，仅有明确价值的少数方向扩大验证。
- **数据在哪里：**公开的 [VLA-Quant](https://github.com/zsure27/VLA-Quant) 保存代码、报告、小结果和导航；完整原始归档保存在私有 [VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives) 的 Release 附件。位置、逐项核验进度和[恢复命令](backups/github/RESTORE_CN.md)见[云端归档索引](backups/github/README_CN.md)与[备份策略](docs/GITHUB_BACKUP_POLICY_20261017_CN.md)。

本次私有归档迁移和已授权原始备份副本清理已完成。完整恢复目录、下载校验和清理回执见[云端归档索引](backups/github/README_CN.md)；本机保留工作文件、小结果和恢复入口，大型历史材料按需恢复。B5只有历史部分区间，不能据此恢复完整包。审计缓存与旧附件整理也已完成，见[私有完成摘要](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261019/migration-completion.json)。

091 已取得原生关机 `execute=true` 回执，心跳暂停。B1 时间覆盖实验以正式各 **250** 回合收束；中断的 reset30–39 残余保留并排除，不恢复旧续片。本仓库的整理不启动实例或实验。

## 最近三组对照

| 问题 | 结果 | 当前认识 |
|---|---|---|
| 相同训练预算下，普通零输出初始化与 Response-SVD 初始化有何差别？ | 普通初始化 Z：2/450；SVD 初始化 S：426/450 | 当前配方对初始化高度敏感；SVD 额外校准成本需单列 |
| SVD 初始化在不训练时是否已经有效？ | PTQ：77/450；S0：184/450，119 救回、12 新增失败 | 初始化本身净提高 23.78 个百分点，但恢复仍不充分 |
| 同样 80 个训练观测，取帧时间是否重要？ | Early80：241/250；Phase80：110/250，Phase 净低 52.4 个百分点 | 当前 Phase 取帧方案明显较差；原因需要动作和数据诊断 |

以上是三组不同运行，分母不合并，也不跨轮直接相减计算训练收益。当前结果属于同十任务开发证据；重叠微测和中断残余不增加正式分母。训练误差下降不能单独证明闭环收益，fake quant 结果不能作为 INT2/INT4 实际部署加速结论。

## 仓库导航

| 入口 | 内容 |
|---|---|
| [reports/](reports/README_CN.md) | 实验分析、综合总结和历史报告 |
| [results/](results/README_CN.md) | 小型原始结果、命令、配对数据、门禁和 SHA 清单 |
| [backups/](backups/README_CN.md) | 完整归档位置、云端资产索引和恢复约定 |
| [qvla/](qvla/README_CN.md) | 量化、校准、微调与运行时实现 |
| [diagnostics/](diagnostics/README_CN.md) | 机制诊断和结果审计 |
| [scripts/](scripts/README_CN.md) | 执行、备份、恢复与收尾工具 |
| [模型身份与数据审计](docs/MODEL_NAMING_AND_DATA_AUDIT_20261005_CN.md) | 历史模型别名、数据角色与来源约束 |

复核历史结果时使用其冻结源码、参数、输入和 SHA，不用当前维护代码改写历史证据。A0 为官方语言 AWQ 加项目视觉适配参考，尚不能称为 QVLA 通道分配的严格复现。凭据、私钥和令牌不进入仓库或归档。

整理前 README 的[完整原字节快照](docs/archive/README_BEFORE_CLEANUP_20261017.md)保留历史说明；其中相对路径按原仓库根目录解释，旧活动状态不代表当前待办。布局和历史迁移见[仓库说明](docs/REPOSITORY_LAYOUT_20260925_CN.md)与[CHANGELOG](CHANGELOG_CN.md)。
