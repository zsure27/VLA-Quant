# VLA 快速量化与微调：个人实验备份

保存 OpenVLA-OFT 低比特量化的代码、配置、日志、结果数据与分析，用于复核已完成实验和恢复后续运行。最新实测截止北京时间 **2026-09-29**。

后续每项实验在启动前先执行[研究方向自纠偏门禁](docs/VLA_RESEARCH_GOVERNOR_20260926.txt)，生成可核对的 pre-run card；结束后保存 post-run decision，再决定下一项。[当前阶段纠偏决定](docs/RESEARCH_DIRECTION_CORRECTION_20260927_CN.md)解释了 P2.5 的因果问题与门禁。仓库根目录 `AGENTS.md` 将此设为持续工作规则。

运行中的实例使用服务器顺序执行器和约 15 分钟的低频静默心跳；阶段结束及时接续，收尾关机后暂停心跳。不得靠反复唤醒对话计时。完整约定见[自适应实验与备份分析协议](docs/ADAPTIVE_EXPERIMENT_AND_BACKUP_PROTOCOL_20260925_CN.md)和[运行入口](scripts/README_CN.md)。

长闭环评测在执行前还必须通过[低成本烟雾门禁](docs/EVALUATION_SMOKE_GATE_20260929_CN.md)：真实评测顺序的无模型重复探针、配对策略微型闭环、首片复核依次通过后才扩展长片。

后续 A1/A2 复现按[最小 C0/C3 配对与 rescue/break 协议](docs/A1_A2_MINIMAL_REPLICATION_AND_CONFLICT_PROTOCOL_20260929_CN.md)执行；C1 不再重跑，路由判断还需同任务可观测上下文证据。

[2026-09-30 自查与下次开机执行卡](docs/NEXT_BOOT_A1_A2_AUDIT_AND_RUNBOOK_20260930_CN.md)补齐真实模型输入、训练随机源和统计门禁；没有合格的新评测条件时先做低成本审计，不派发长闭环。

## 实验记录入口

最新综合报告：[2026-09-29 基线进展与后续规划](reports/summaries/20260929-baseline-progress/README_CN.md)。它区分静态 AWQ 基线、旧 PEFT 负结果、C3 的单条件收益和 A1 独立性失败。

综合目录 [2026-09-18 实验总结](reports/summaries/20260918-experiment-overview/) 保留三个文件：

- [结论与研究计划](reports/summaries/20260918-experiment-overview/README_CN.md)：图表、问题分析和后续条件。
- [按实验名称整理的日志](reports/summaries/20260918-experiment-overview/EXPERIMENT_LOG_CN.md)：配置、结果和原始证据入口。
- [证据索引](reports/summaries/20260918-experiment-overview/EVIDENCE_INDEX.json)：权威汇总、来源路径、字节数和 SHA256；原始数值保留在所指文件。

[各轮记录](reports/README_CN.md)保存当轮数据与分析；[原始结果](results/README_CN.md)保存console、命令、manifest和逐回合指标。[精确重复文件索引](results/indexes/README_CN.md)给出移除文件与规范副本的 SHA256 映射。较早的[分析归档](reports/archive/20260916-weekend-review/)保留历史证据，旧状态不覆盖最新结论。

## 当前结果

**2026-09-29 更新：**学生状态重标的 C3 rank8 Recovery-LoRA 在 12L 混精底座、未参与该候选训练的开发初态5–49上得到 `389/450`，同条件 C0 为 `368/450`、静态14L为 `388/450`；任务1/5收益集中，任务3/6/7退步。随后110的 `env_seed=1` 重跑虽得到相同数字，但真实评测顺序下没有建立新的策略可见条件，判为 `A1_INVALID_CONDITION`，**不计独立复现**。详见[107的C3结果](reports/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/POST_RUN_ANALYSIS_CN.md)与[110的A1审计](reports/experiments/p2-shared-peft/20260929-110-a1-envseed1/README_CN.md)。下表中的旧 data80 LoRA 结果为不同训练方案的历史负基线，不能覆盖 C3，也不能与新候选的450回合相加。

下一阶段先提高 12L/C3 并检验与 BF16 的同条件差距，纯 W2 后续再迁移；[恢复路线](docs/C3_TO_BF16_RECOVERY_ROADMAP_20260929_CN.md)明确了简单 C0/C3 路由的事后空间和进入 Router 的证据门槛。[下一轮 LoRA 扩展实验卡](docs/LANGUAGE_VISION_W2_LORA_EXPANSION_PLAN_20260930_CN.md)按语言扩展、增补视觉、全目标 W2 迁移逐级设置配对基线与烟雾门禁。

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
| 14层静态候选 | 431/500 | 本轮与12L/LoRA同评测器配对；另一完整复跑430，来源审计中；W4 blocks 8–15、18–23 |
| 12层PEFT底座 | 411/500 | 本轮配对；历史412，来源审计中；blocks18–19保持W2 |
| 12层+data80 rank8 Recovery-LoRA | 408/500 | 本轮配对；相对12L救回25回合、新增失败28回合，净闭环恢复未成立 |

SQ W4A4尚未验收。Scale-PEFT 虽显著降低固定观测误差，但完整闭环为407/500，现作为负基线。rank8 Response-SVD Recovery-LoRA 的 data80 版本同时改变轨迹覆盖16→80和优化步数200→1000，不能单独归因于覆盖。轨迹级 A/B/C/D 已补齐，增加训练步数与覆盖均未呈单调收益。本轮 500 回合中 LoRA 的动作拟合改善没有形成净闭环收益；在其学生访问的 500 个同观测上，LoRA 比无 LoRA 更接近 BF16 的样本为421个，但同状态探针并非无 LoRA 的反事实闭环。下一步是 P2.5 的失败时序与目标错配诊断，专家和 Router 继续锁定。当前仍是BF16存储的fake quant，不据此报告INT2/INT4实际压缩与加速。

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
[2026-09-22 计划](docs/PEFT_CONTEXTUAL_ROUTING_PLAN_20260922_CN.md)。结合 2026-09-28 配对结果的研究问题、阶段门禁与条件分支见[低比特恢复与 Contextual Routing 修订规划](docs/VLA_QUANT_CONTEXTUAL_RECOVERY_ROADMAP_20260928_CN.md)。技术变更见[CHANGELOG](CHANGELOG_CN.md)；仓库整理见[2026-09-22 记录](docs/REPOSITORY_CLEANUP_20260922_CN.md)和[2026-09-30 记录](docs/REPOSITORY_CLEANUP_20260930_CN.md)。
