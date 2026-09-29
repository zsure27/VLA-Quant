# 2026-09-30 仓库整理记录

本次只整理本地 Git 文件树，没有连接付费服务器或产生新的 GPU 结果。清理前共 1,910 个跟踪文件；`reports/experiments/` 和 `results/experiments/` 已按 P0–P5/次线模块分组，因此保留既有会话名和原始测量。

## 处理内容

| 类型 | 处理 | 可追溯性 |
|---|---|---|
| 014 静态裁剪会话的 66 份评测器/干预源码副本 | 三种源码各 22 份逐字节相同；移除逐分片副本，保留 3 份 SHA 命名快照（其中 OFT 快照原已存在） | [`精确映射`](../results/indexes/DEDUPLICATED_FILES_20260930.json)逐项保存旧路径、字节数与 SHA256；会话 manifest 的 `original_path` 保留原名，`path` 指向当前快照 |
| 纯预注册 A1 卡片 | 从 `reports/experiments/` 移至 [`docs/pre-run-cards/`](pre-run-cards/) | 没有对应的实测结果，移出后不会伪装成已执行会话；059 预检报告已更新链接，卡片顶部注明 A1 失效 |
| 旧 Response-SVD 归档工具 | 从 Git 历史恢复当时匹配 SHA 的源码快照，替换 manifest 中指向已演进维护脚本的路径 | 原脚本路径、Git blob 与 SHA 仍在 manifest 中；当前脚本不被回退 |
| 目录与核验入口 | 新增 `results/shared/`、`results/indexes/` 和预注册卡片索引；校验器支持全部去重索引、定点会话校验和本地 Python 3.7 | [`仓库布局`](REPOSITORY_LAYOUT_20260925_CN.md)和各目录 README 说明边界 |

66 份删除副本合计 **1,256,992 字节**；两份新增规范快照共 35,128 字节，OFT 快照本来已存在，因此当前文件树约净省 **1,221,864 字节**。这仅是工作树/新提交的净变化；历史 Git blob 仍保留，不能据此声称整个 Git 历史同量缩小。

逐回合结果、`policy-queries.jsonl`、不同的配置/指标、`CONTRACT_SHA256SUMS.txt`、退出码、原始视频清单和完整本地/服务器归档均未删除。`probe110.json` 与字节相同的 repeat 文件承担重复性审计，不按普通重复内容删除。历史 manifest 中的服务器绝对路径不改写；新索引仅描述当前 Git 文件树的规范位置。

## 验证

在仓库根目录执行：

```text
python scripts/verify_deduplicated_results.py
python scripts/verify_session_manifests.py
python scripts/validate_repository_layout.py
python scripts/verify_repository.py
```

去重索引同时核对每个被删除旧路径不存在、规范副本存在且 SHA/字节数相同；会话清单核验报告、结果及共享快照。静态裁剪会话仍为 22 分片、580 回合和 10,020 次策略查询，本次整理不重新计算成功率，也不改写历史研究结论。
