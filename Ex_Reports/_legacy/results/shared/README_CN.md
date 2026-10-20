# 共享不可变证据

本目录存放多个实验共同引用、且字节已校验相同的材料；它不是新的实验结果，也不能把不同会话视为独立样本。

| 目录 | 内容 | 使用规则 |
|---|---|---|
| `source-snapshots/` | 按 SHA 命名的评测器和干预源码快照 | 按报告 manifest 的 SHA 复核，不以当前维护代码替代历史版本 |
| `calibration/` | 完全相同的共享校准统计 | 仍以各会话配置、来源和切分记录解释 |
| `historical/` | 早期汇总 | 标记为历史证据，不并入正式配对闭环 |

历史分片中移除的字节相同副本及其原路径，见 [`2026-09-22 映射`](../indexes/DEDUPLICATED_FILES_20260922.json)与[`2026-09-30 映射`](../indexes/DEDUPLICATED_FILES_20260930.json)。每份映射列出保留路径、SHA256、字节数；`python scripts/verify_deduplicated_results.py` 可从仓库根目录核验。旧布局可从对应 Git 历史或完整会话归档恢复。

`backup-active-diagnostics-eea5abdcc493.py` 是 2026-09-17 Response-SVD 会话时的归档工具源码快照，从 Git blob `ad0c97f04130ace71003953d0ed2566654d9146f` 恢复。当前 `scripts/backup_active_diagnostics.py` 已演进，不能用其当前字节冒充旧会话源码；旧路径和 blob ID 记录在该会话 manifest 的 `archival_source_replacement` 中。
