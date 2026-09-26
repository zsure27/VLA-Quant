# VLA 快速量化与微调：个人实验备份

保存 OpenVLA-OFT 低比特量化的代码、配置、日志、结果数据与分析，用于复核已完成实验和恢复后续运行。最新实测截止北京时间 **2026-09-26**。

## 实验记录入口

综合目录 [2026-09-18 实验总结](reports/summaries/20260918-experiment-overview/) 保留三个文件：

- [结论与研究计划](reports/summaries/20260918-experiment-overview/README_CN.md)：图表、问题分析和后续条件。
- [按实验名称整理的日志](reports/summaries/20260918-experiment-overview/EXPERIMENT_LOG_CN.md)：配置、结果和原始证据入口。
- [证据索引](reports/summaries/20260918-experiment-overview/EVIDENCE_INDEX.json)：权威汇总、来源路径、字节数和 SHA256；原始数值保留在所指文件。

[各轮记录](reports/README_CN.md)保存当轮数据与分析；[原始结果](results/README_CN.md)保存console、命令、manifest和逐回合指标。[精确重复文件清单](results/indexes/DEDUPLICATED_FILES_20260922.json)给出移除文件与规范副本的 SHA256 映射。较早的[分析归档](reports/archive/20260916-weekend-review/)保留历史证据，旧状态不覆盖最新结论。

## 当前结果

| 配置 | 成功回合 | 评估范围 |
|---|---:|---|
| BF16 | 487/500 | Spatial，官方初态0–49 |
| AWQ W4A16 | 486/500 | 与BF16严格配对 |
| 仅视觉AWQ W2，DINO64/SigLIP128 | 460/500 | 语言BF16，同500回合 |
| 仅语言W2 G64、注意力撤销额外clip | 开发29/50；扩展49/100 | 初态5–9与10–19分开记录 |
| 完整连接目标W2，原始G128 | 0/50 | 初态5–9，保护模块BF16 |
| 完整连接目标W2，G64注意力无clip | 11/50 | 同50回合，视觉也是G64 |
| 完整W2视觉group归因（语言固定G64注意力无clip） | DINO128/SigLIP128 3/50；DINO128/SigLIP64 1/50；DINO64/SigLIP128 10/50；DINO64/SigLIP64 11/50 | 初态5–9严格配对；主要恢复来自DINO G64 |
| 完整W2底座加语言W4岛 | 8–15层27/50；8–23层44/50；16–23层短筛13/20 | DINO64/SigLIP128固定；属于混合W2/W4 |
| 16层静态参照 | 434/500 | W4 blocks 8–23；较强静态恢复参考 |
| 14层静态候选 | 约430/500 | 历史431、最新完整复跑430，来源审计中；W4 blocks 8–15、18–23 |
| 12层PEFT底座 | 约411/500 | 历史412、最新完整复跑411，来源审计中；blocks18–19保持W2 |

SQ W4A4尚未验收。Scale-PEFT 虽显著降低固定观测误差，但完整闭环为407/500，低于约
411/500 的12L，现作为负基线。rank8 Response-SVD Recovery-LoRA 的 data80 Smooth-L1 版本
在历史32帧上降至0.02284、31/32改善，但它同时把轨迹数16→80、优化步200→1000，不能归因
为单独的覆盖率收益。当前50回合slice的12L与14L均为43/50，因此 data80 闭环只作为稳定性
试跑，不能计算W4恢复率。后续已改为轨迹级router_dev、A/B/C/D compute-vs-coverage与P2.5
学生访问观测上的BF16同观测诊断，尚未进入专家或Router。当前仍是BF16存储的fake quant，
不据此报告INT2/INT4实际压缩与加速。

当前 PEFT / Contextual Routing 唯一默认底座为 [`awq-w2a16-12l-mixed-spatial-v1`](configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)：DINO W2 G64、SigLIP W2 G128，语言 W4 blocks 8–15 与 20–23，其余语言 W2 G64。第一版 PEFT 只作用于仍为 W2 的 blocks 18–19。14L 和 16L 只承担静态恢复参照；若新证据要求改变底座，新增版本化配置并保留旧版本。

## 目录

| 路径 | 内容 |
|---|---|
| [`reports/`](reports/README_CN.md) | 按 P0–P5/次线模块归档的分析、图表、汇总与历史报告 |
| [`results/`](results/README_CN.md) | 与报告同名分组的原始小日志、数据、配置、哈希及共享快照 |
| [`backups/`](backups/README_CN.md) | 本地完整归档的统一入口；`backups/experiments/` 不进入普通 Git |
| [`qvla/`](qvla/README_CN.md) | 维护中的量化、校准、评估与运行时契约代码 |
| [`diagnostics/`](diagnostics/README_CN.md) | 机制诊断与结果审计代码 |
| [`scripts/`](scripts/README_CN.md) | 有限实验分片、恢复、备份与收尾工具 |
| `configs/`、`data/` | 固定版本、422目标、数据来源及大文件恢复说明 |
| `overlays/`、`legacy/` | OFT覆盖文件及历史实现，供复核旧结果 |
| `docs/` | 技术说明、安装修复与研究计划 |
| `tests/` | 数值及实现契约检查，不替代GPU闭环验收 |

复核某次结果时使用其记录的 checkpoint、profile、代码与输入哈希，不用当前维护代码改写历史测量。字节相同的源码快照归并到 `results/shared/source-snapshots/`，历史路径和哈希见精确重复文件清单；不同版本仍留在原分片。

## 恢复与维护

环境恢复见[安装修复](docs/INSTALL_REPAIR_20260911_CN.md)与[基线审计](docs/BASELINE_AUDIT_20260908_CN.md)。`scripts/bootstrap_autodl.sh`默认固定源码和适配代码，不自动下载大模型或启动GPU实验。已停用的v2批处理只保留提示入口。

综合数据与摘要图可在本地重建：

```bash
python scripts/build_experiment_summary.py
```

模型、校准集、视频、NPZ和大 profile 不放普通 Git，恢复位置及 SHA 以各轮清单为准。双份归档不代表所有模型和数据均已异地备份。凭据和私钥不入仓库。目录命名、模块边界和旧路径映射见[2026-09-25 仓库布局说明](docs/REPOSITORY_LAYOUT_20260925_CN.md)。

静态 W4 block 组合搜索已冻结。后续以 12L 为同一低比特底座，在 blocks18–19 先完成
Recovery-LoRA 的 compute-vs-coverage 与 P2.5 闭环对齐诊断；固定观测误差下降不再作为进入
专家阶段的充分证据。当前修正见
[2026-09-26 P2.5 计划](docs/P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md)，整体执行顺序见
[2026-09-25 主线协议](docs/PEFT_CONTEXTUAL_ROUTING_EXECUTION_20260925_CN.md)，结果驱动的调整、
历史实验教训和“数据+分析”备份验收见
[自适应实验与备份分析协议](docs/ADAPTIVE_EXPERIMENT_AND_BACKUP_PROTOCOL_20260925_CN.md)，证据设计见
[2026-09-22 计划](docs/PEFT_CONTEXTUAL_ROUTING_PLAN_20260922_CN.md)。技术变更见[CHANGELOG](CHANGELOG_CN.md)，本次清理见[仓库整理记录](docs/REPOSITORY_CLEANUP_20260922_CN.md)。
