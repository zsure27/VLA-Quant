# VLA 快速量化与微调：个人实验备份

保存 OpenVLA-OFT 低比特量化的代码、配置、日志、结果数据与分析，用于复核已完成实验和恢复后续运行。最新实测截止北京时间 **2026-09-25**。

## 实验记录入口

综合目录 [2026-09-18 实验总结](reports/20260918-vla-experiment-summary/) 保留三个文件：

- [结论与研究计划](reports/20260918-vla-experiment-summary/README_CN.md)：图表、问题分析和后续条件。
- [按实验名称整理的日志](reports/20260918-vla-experiment-summary/EXPERIMENT_LOG_CN.md)：配置、结果和原始证据入口。
- [证据索引](reports/20260918-vla-experiment-summary/EVIDENCE_INDEX.json)：权威汇总、来源路径、字节数和 SHA256；原始数值保留在所指文件。

[各轮记录](reports/sessions/README_CN.md)保存当轮数据与分析；[原始结果](results/README_CN.md)保存console、命令、manifest和逐回合指标。[精确重复文件清单](results/DEDUPLICATED_FILES_20260922.json)给出移除文件与规范副本的 SHA256 映射。较早的[分析归档](reports/2026-09-16-weekend-review-archive/)保留历史证据，旧状态不覆盖最新结论。

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
| 14层静态候选 | 431/500 | W4 blocks 8–15、18–23；部署膝点候选 |
| 12层PEFT底座 | 412/500 | W4 blocks 8–15、20–23；blocks18–19保持W2 |

SQ W4A4尚未验收；PEFT已有低秩初始化测量，但没有梯度训练、LoRA闭环或Router结果。当前是BF16存储的fake quant，不据此报告INT2/INT4实际压缩与加速。完整四套件、多种子及真实kernel评测未完成。

## 目录

| 路径 | 内容 |
|---|---|
| `reports/` | 综合分析、逐轮记录、图表与历史归档 |
| `results/` | 原始小日志、数据、命令、配置、哈希及源码快照 |
| [`qvla/`](qvla/README_CN.md) | 维护中的量化、校准、评估与运行时契约代码 |
| [`diagnostics/`](diagnostics/README_CN.md) | 机制诊断与结果审计代码 |
| [`scripts/`](scripts/README_CN.md) | 有限实验分片、恢复、备份与收尾工具 |
| `configs/`、`data/` | 固定版本、422目标、数据来源及大文件恢复说明 |
| `overlays/`、`legacy/` | OFT覆盖文件及历史实现，供复核旧结果 |
| `docs/` | 技术说明、安装修复与研究计划 |
| `tests/` | 数值及实现契约检查，不替代GPU闭环验收 |

复核某次结果时使用其记录的 checkpoint、profile、代码与输入哈希，不用当前维护代码改写历史测量。字节相同的源码快照归并到 `results/source-snapshots/`，历史路径和哈希见精确重复文件清单；不同版本仍留在原分片。

## 恢复与维护

环境恢复见[安装修复](docs/INSTALL_REPAIR_20260911_CN.md)与[基线审计](docs/BASELINE_AUDIT_20260908_CN.md)。`scripts/bootstrap_autodl.sh`默认固定源码和适配代码，不自动下载大模型或启动GPU实验。已停用的v2批处理只保留提示入口。

综合数据与摘要图可在本地重建：

```bash
python scripts/build_experiment_summary.py
```

模型、校准集、视频、NPZ和大 profile 不放普通 Git，恢复位置及 SHA 以各轮清单为准。双份归档不代表所有模型和数据均已异地备份。凭据和私钥不入仓库。此次整理保留全部唯一的逐回合日志和指标，仅移除哈希核对过的重复文件及汇总 JSON 的重复数值副本；历史 Git 提交仍可取回原目录布局。

静态 W4 block 组合搜索已冻结。后续以 12L 为同一低比特底座，先完成训练数据与 q/z/Δ 契约，再在 blocks18–19 做 shared Scale-PEFT 与 Recovery LoRA；只有 shared PEFT 闭环有效且两个等预算专家呈现稳定互补后，才训练基于 block17 后因果表示的 Top-1 Router。执行顺序见[2026-09-25 主线协议](docs/PEFT_CONTEXTUAL_ROUTING_EXECUTION_20260925_CN.md)与[2026-09-22 证据计划](docs/PEFT_CONTEXTUAL_ROUTING_PLAN_20260922_CN.md)。技术变更见[CHANGELOG](CHANGELOG_CN.md)，本次清理见[仓库整理记录](docs/REPOSITORY_CLEANUP_20260922_CN.md)。
