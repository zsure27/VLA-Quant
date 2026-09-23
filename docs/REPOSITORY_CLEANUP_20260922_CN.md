# 2026-09-22 仓库整理记录

本次只整理本地 `VLA-Quant` 的 Git 文件树，没有连接 AutoDL、生成 GPU 结果或改写既有实验结论。整理基于原提交 `5c034a84b1cd0fda7e09aac1166b6af399064b10`；该提交及服务器会话归档仍可恢复原目录布局。

## 移除与保留

| 项目 | 处理 | 核验 |
|---|---|---|
| 50 份字节完全相同的统计与源码副本 | 旧分片路径移除，留下 4 份 SHA 命名规范副本 | 原始字节 SHA256 逐项相等；[映射清单](../results/DEDUPLICATED_FILES_20260922.json)记录全部旧路径、哈希和字节数 |
| 综合报告的重复数值副本 | `结果数据.json` 改为 `EVIDENCE_INDEX.json`，只索引原件并保留权威汇总 | 源文件仍在 `results/` 与 `reports/sessions/`；114 份来源与 1000 条配对 scope 记录仍被生成器核对 |
| 历史 `SHA256SUMS.txt` | 保留原文，不把整理后的 Git 路径冒充服务器归档原路径 | 当前会话 manifest 重新索引 Git 文件树；历史回执用于验证原始归档 |
| 唯一原始数据 | 保留逐回合 EVAL/manifest、`policy-queries.jsonl`、不同的 `metrics.json`、唯一诊断统计、profile 来源清单和收尾回执 | 未按文件名或体积删除唯一记录；未触碰 `.gitignore` 排除的 `results/pending-*` 本地归档 |

精确重复文件原计 31,644,254 字节，规范副本计 7,740,350 字节，当前树净省 **23,903,904 字节**。综合 JSON 原计 6,932,740 字节，证据索引重建后约 83 KB，另外减少约 6.85 MB。Git 历史保留旧 blob，普通整理提交不会缩小既有仓库历史对象；这保证原测量仍可追溯。

## 目录和代码入口

- 综合报告统一为 [`reports/20260918-vla-experiment-summary/`](../reports/20260918-vla-experiment-summary/)：`README_CN.md` 是结论，`EXPERIMENT_LOG_CN.md` 是按实验名称的证据索引，`EVIDENCE_INDEX.json` 是机器可读的哈希清单。
- 报告生成器改名为 [`scripts/build_experiment_summary.py`](../scripts/build_experiment_summary.py)。生成器不连接服务器、不补造数据，图表只由已提交的测量汇总产生。
- [`qvla/README_CN.md`](../qvla/README_CN.md)、[`diagnostics/README_CN.md`](../diagnostics/README_CN.md)、[`scripts/README_CN.md`](../scripts/README_CN.md)按职责标明当前代码、诊断和运维入口。旧 v2 拒绝执行脚本继续留在原位置，以覆盖服务器上可能遗留的危险旧命令。
- `results/source-snapshots/` 保存不可变源码快照；`results/shared-calibration/` 保存共享统计。不同版本源码与统计留在原实验目录。

## 整理后核验

可在仓库根目录运行：

```text
python scripts/verify_deduplicated_results.py
python scripts/verify_session_manifests.py
python reports/sessions/20260917-107-baseline-continuation/manifest.py --verify
python reports/sessions/20260918-107-awq-continuation/manifest.py --verify
python scripts/build_experiment_summary.py
```

其中综合报告生成器会断言正式 BF16/W4 为 487/500、486/500，视觉 W2 为 460/500，完整 W2 两配方为 0/50 和 11/50，语言注意力方案两批为 29/50、49/100；它还检查主报告的相对链接。整理不改变这些数值。
