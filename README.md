# VLA 快速量化与微调：个人实验备份

保存 OpenVLA-OFT 低比特量化的代码、配置、日志、结果数据与分析，用于复核已完成实验和恢复后续运行。最新实测截止北京时间 **2026-09-18 03:37**。

## 实验记录入口

综合目录 [reports/2026-09-16-weekend-review](reports/2026-09-16-weekend-review/) 保留三个文件：

- [汇报总结](reports/2026-09-16-weekend-review/汇报总结.md)：图表、问题分析、改进效果、PEFT/Contextual Routing与规划。
- [实验日志](reports/2026-09-16-weekend-review/实验日志.md)：按实验名称记录配置、结果及原始日志入口。
- [结果数据](reports/2026-09-16-weekend-review/结果数据.json)：表格、指标、配对结果、来源路径和SHA；大型逐层统计保留原文件索引。

[各轮记录](reports/sessions/README_CN.md)保存当轮数据与分析；[原始结果](results/README_CN.md)保存console、命令、manifest、指标和实验时源码。较早的[分析归档](reports/2026-09-16-weekend-review-archive/)保留历史证据，旧状态不覆盖最新结论。

## 当前结果

| 配置 | 成功回合 | 评估范围 |
|---|---:|---|
| BF16 | 487/500 | Spatial，官方初态0–49 |
| AWQ W4A16 | 486/500 | 与BF16严格配对 |
| 仅视觉AWQ W2，DINO64/SigLIP128 | 460/500 | 语言BF16，同500回合 |
| 仅语言W2 G64、注意力撤销额外clip | 开发29/50；扩展49/100 | 初态5–9与10–19分开记录 |
| 完整连接目标W2，原始G128 | 0/50 | 初态5–9，保护模块BF16 |
| 完整连接目标W2，G64注意力无clip | 11/50 | 同50回合，视觉也是G64 |

SQ W4A4尚未验收；PEFT已有低秩初始化测量，但没有梯度训练、LoRA闭环或Router结果。当前是BF16存储的fake quant，不据此报告INT2/INT4实际压缩与加速。完整四套件、多种子及真实kernel评测未完成。

## 目录

| 路径 | 内容 |
|---|---|
| `reports/` | 综合分析、逐轮记录、图表与历史归档 |
| `results/` | 原始小日志、数据、命令、配置、哈希及源码快照 |
| `qvla/`、`diagnostics/` | 量化适配、校准、评估与机制诊断 |
| `scripts/` | 有限实验分片、恢复、备份与收尾工具 |
| `configs/`、`data/` | 固定版本、422目标、数据来源及大文件恢复说明 |
| `overlays/`、`legacy/` | OFT覆盖文件及历史实现，供复核旧结果 |
| `docs/` | 技术说明、安装修复与研究计划 |
| `tests/` | 数值及实现契约检查，不替代GPU闭环验收 |

复核某次结果时使用其记录的checkpoint、profile、代码与输入哈希，不用当前维护代码改写历史测量。源码快照用于实验版本复核；重复图表保留原始轮次副本。

## 恢复与维护

环境恢复见[安装修复](docs/INSTALL_REPAIR_20260911_CN.md)与[基线审计](docs/BASELINE_AUDIT_20260908_CN.md)。`scripts/bootstrap_autodl.sh`默认固定源码和适配代码，不自动下载大模型或启动GPU实验。已停用的v2批处理只保留提示入口。

综合数据与摘要图可在本地重建：

```bash
python scripts/build_weekend_review.py
```

模型、校准集、视频、NPZ和大profile不放普通Git，恢复位置及SHA以各轮清单为准。双份归档不代表所有模型和数据均已异地备份。凭据和私钥不入仓库。仅清理确认冗余的派生文件，保留唯一原件。

后续先验证AWQ完整W2的语言/视觉组合损失，SQ数值控制其次；静态PEFT有效且专家互补可泛化后再考虑路由。技术变更见[CHANGELOG](CHANGELOG_CN.md)，最近整理清单见[维护记录](docs/REPOSITORY_CLEANUP_20260918.json)。