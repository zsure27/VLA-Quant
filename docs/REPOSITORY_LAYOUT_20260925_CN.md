# VLA-Quant 实验目录与命名规范

更新：2026-09-30。

## 1. 三类资产

| 位置 | 用途 | 是否进入普通 Git |
|---|---|---|
| `reports/experiments/<module>/<session>/` | 结论、图表、分析数据、manifest、复现入口 | 是 |
| `results/experiments/<module>/<session>/` | 命令、console、配置、逐回合小指标、源码和哈希 | 是 |
| `backups/experiments/<module>/<session>/` | bundle、patch、环境锁、压缩完整结果、视频与大文件清单 | 否 |

`<session>` 统一为 `YYYYMMDD-<host-or-scope>-<experiment-slug>`。同一实验的 report 与 result 名称必须相同；完整归档可以在会话目录下再按 archive ID 分层。

## 2. 模块

| 模块 | 含义 | 当前状态 |
|---|---|---|
| `p0-foundation-baselines` | 基线、W2 失效与归因 | 已完成 |
| `p1-data-contract` | 训练轨迹、切分、泄漏和参数化契约 | 已建立，持续审计 |
| `p2-shared-peft` | 12L backbone blocks 18–19 的 shared PEFT | C3 有单条件收益；A1 独立性失败 |
| `p3-static-backbone` | 形成 12L/14L/16L 的静态搜索 | 冻结 |
| `p4-expert-complementarity` | 等预算专家及互补矩阵 | 条件待做 |
| `p5-contextual-routing` | H17 Top-1 Router | 条件待做 |
| `secondary-smoothquant` | SQ 数值与闭环扩展 | 主线验证后再做 |

## 3. Backbone 版本

当前唯一默认 PEFT/Router backbone 是 [`awq-w2a16-12l-mixed-spatial-v1`](../configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)。配置变化必须产生新 ID 和新文件，并记录证据、配对结果及替换理由。历史配置和证据目录不覆盖、不重命名为新含义。

## 4. 自动检查

`scripts/validate_repository_layout.py` 检查模块集合、会话命名、报告与结果配对、废弃顶层目录、主 backbone 字段和 JSON 可解析性。收尾脚本要求显式提供 `-ExperimentModule`，并把完整本地归档写入统一备份目录。

2026-09-30 将 014 静态裁剪会话中的 66 份字节相同源码副本归并为 3 份 SHA 命名的共享快照。该会话 `manifest.json` 的 `original_path` 保留取回时的逐分片路径，`path` 指向当前规范副本；映射位于 [`DEDUPLICATED_FILES_20260930.json`](../results/indexes/DEDUPLICATED_FILES_20260930.json)。历史 `CONTRACT_SHA256SUMS.txt` 记录的服务器路径和模型/profile SHA 均未改写。逐分片日志、指标、trace、退出码及完整归档不参与归并。使用 `scripts/verify_deduplicated_results.py` 和 `scripts/verify_session_manifests.py` 核验当前树。

旧路径到当前路径的完整映射见 [`REPOSITORY_LAYOUT_MIGRATION_20260925.json`](REPOSITORY_LAYOUT_MIGRATION_20260925.json)。历史 SHA 回执中的服务器路径保持原样。

仅有预注册卡片、没有任何实测结果的方案位于 [`docs/pre-run-cards/`](pre-run-cards/)，不创建空的 `reports/experiments/` 会话。可执行脚本的原相对路径暂保持稳定，避免破坏历史命令和远端部署；代码职责由 `qvla/`（模型实现）、`diagnostics/`（诊断）与 `scripts/`（执行和运维）各自 README 定义。
