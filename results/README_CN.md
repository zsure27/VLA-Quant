# 原始结果与可复核性

`results/` 按会话保存实际命令、配置、运行日志、逐回合评估和离线诊断指标。早期的 `official_quant_validation.csv` 仅由历史摘要恢复，标为 `archival_summary`；它的 488/500、491/500 与 2026-09-17 冻结协议下的正式 Spatial500（BF16 487/500、AWQ W4 486/500）不能合并。当前结论与证据入口见[综合报告](../reports/20260918-vla-experiment-summary/README_CN.md)和[实验索引](../reports/20260918-vla-experiment-summary/EXPERIMENT_LOG_CN.md)。

`107-awq-p0-20260923/` 保存最新三格视觉 group 归因的命令、console、EVAL 日志、策略 trace、冻结源码和审计回执；配对分析见[会话报告](../reports/sessions/20260923-107-awq-p0-visual-groups/README_CN.md)。

## 目录约定

- `107-*-YYYYMMDD/`：已取回并提交的会话小文件。一个分片中的 `command.txt`、`console.log`、`EVAL-*.txt`、`policy-queries.jsonl`、`metrics.json`、`episode-manifest.json` 等按实际产物保存；不同实验可能缺少部分文件，不补造。
- `source-snapshots/`：多个分片中内容完全相同的评测器与探针源码，只保留一份不可变的 SHA 命名快照。运行时源文件的 SHA 仍记录在各分片的 `CONTRACT_SHA256SUMS.txt` 等凭据中。
- `shared-calibration/`：四份完全相同的动作 token RMS 输入统计只保留一份。另一份不同的全 token 统计仍留在原实验目录。
- [精确重复文件映射](DEDUPLICATED_FILES_20260922.json)：逐项记录旧路径、规范副本、原始字节数和 SHA256。原始 Git 提交 `5c034a8` 可取回清理前的目录布局。旧 `SHA256SUMS.txt` 是服务器归档时的历史回执，不修改其中的原路径；规范副本具有相同字节哈希。

`pending-*` 是取回的完整本地归档，受 `.gitignore` 排除；本次清理未触碰这些文件。模型、原校准集、视频、NPZ 和大 profile 不放普通 Git，恢复路径与 SHA 见各会话 `backup/` 清单。清单本身不等于大文件已全部异地备份。

对比配置时应使用相同的 checkpoint、profile、初态、种子和评测器，不能把开发集重复回合当作新样本。`policy-queries.jsonl` 和唯一的逐帧 `metrics.json` 保留在 Git，供后续误差与闭环归因；综合索引只引用这些原件，不重复嵌入大数组。
