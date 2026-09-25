# VLA-Quant 实验目录与命名规范

更新：2026-09-25。

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
| `p1-data-contract` | 训练轨迹、切分、泄漏和参数化契约 | 下一门禁 |
| `p2-shared-peft` | 12L backbone blocks 18–19 的 shared PEFT | 待训练 |
| `p3-static-backbone` | 形成 12L/14L/16L 的静态搜索 | 冻结 |
| `p4-expert-complementarity` | 等预算专家及互补矩阵 | 条件待做 |
| `p5-contextual-routing` | H17 Top-1 Router | 条件待做 |
| `secondary-smoothquant` | SQ 数值与闭环扩展 | 主线验证后再做 |

## 3. Backbone 版本

当前唯一默认 PEFT/Router backbone 是 [`awq-w2a16-12l-mixed-spatial-v1`](../configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)。配置变化必须产生新 ID 和新文件，并记录证据、配对结果及替换理由。历史配置和证据目录不覆盖、不重命名为新含义。

## 4. 自动检查

`scripts/validate_repository_layout.py` 检查模块集合、会话命名、报告与结果配对、废弃顶层目录、主 backbone 字段和 JSON 可解析性。收尾脚本要求显式提供 `-ExperimentModule`，并把完整本地归档写入统一备份目录。

旧路径到当前路径的完整映射见 [`REPOSITORY_LAYOUT_MIGRATION_20260925.json`](REPOSITORY_LAYOUT_MIGRATION_20260925.json)。历史 SHA 回执中的服务器路径保持原样。
