# 结果索引

本目录只放小型机器可读索引；真实测量留在 `results/experiments/`，分析留在同名 `reports/experiments/`。索引不增加独立实验样本数。

- [`DEDUPLICATED_FILES_20260922.json`](DEDUPLICATED_FILES_20260922.json)：早期精确重复统计和源码副本的旧路径、规范副本与 SHA256。
- [`DEDUPLICATED_FILES_20260930.json`](DEDUPLICATED_FILES_20260930.json)：014 静态裁剪会话 66 份重复源码的旧路径、规范快照与 SHA256。

在仓库根目录运行 `python scripts/verify_deduplicated_results.py` 可核验两个索引。历史清单所列旧路径仍可从 Git 历史或完整会话归档恢复。
