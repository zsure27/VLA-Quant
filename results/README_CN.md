# 原始结果与可复核性

`results/` 只保存适合普通 Git 的实测小文件。实验结果统一进入 [`experiments/`](experiments/README_CN.md)，共享不可变材料进入 [`shared/`](shared/)，索引进入 [`indexes/`](indexes/)。完整压缩归档、视频、NPZ、模型和大 profile 保存在本地 [`backups/experiments/`](../backups/README_CN.md)及服务器持久盘。

## 固定结构

```text
results/
  experiments/<module>/YYYYMMDD-<host-or-scope>-<experiment>/
  shared/calibration/
  shared/source-snapshots/
  shared/historical/
  indexes/
```

- [`p0-foundation-baselines`](experiments/p0-foundation-baselines/README_CN.md)：正式基线和量化归因。
- [`p1-data-contract`](experiments/p1-data-contract/README_CN.md)：训练数据、切分和参数化契约。
- [`p2-shared-peft`](experiments/p2-shared-peft/README_CN.md)：shared PEFT 与初始化实验。
- [`p3-static-backbone`](experiments/p3-static-backbone/README_CN.md)：已冻结的静态低比特底座搜索。
- [`p4-expert-complementarity`](experiments/p4-expert-complementarity/README_CN.md)：等预算专家与互补矩阵。
- [`p5-contextual-routing`](experiments/p5-contextual-routing/README_CN.md)：Router 数据、预测和切换 trace。
- [`secondary-smoothquant`](experiments/secondary-smoothquant/README_CN.md)：独立的 SmoothQuant 次线。

一个会话目录可以包含 `command.txt`、`console.log`、`exit-code.txt`、`config.json`、`source-version.json`、`metrics.json`、`episode-manifest.json` 和 `LARGE_FILES_NOT_IN_GIT.json`；实际未生成的项目保持缺失，不补造。

早期 [`official_quant_validation.csv`](shared/historical/official_quant_validation.csv) 是 `archival_summary`，不能与冻结协议下的 Spatial500 合并。[精确重复文件映射](indexes/DEDUPLICATED_FILES_20260922.json)保留旧路径，并把 `replacement` 指向当前规范副本。原始 Git 提交 `5c034a8` 和[布局迁移表](../docs/REPOSITORY_LAYOUT_MIGRATION_20260925.json)可用于恢复旧目录。

对比配置时使用相同 checkpoint、profile、初态、种子和评测器。开发集重复回合不增加独立样本量；离线 MSE 不替代闭环成功率。
