# GitHub 归档与恢复索引

| 仓库 | 保存内容 | 获取方式 |
|---|---|---|
| 公开 [zsure27/VLA-Quant](https://github.com/zsure27/VLA-Quant) | 代码、配置、分析、小结果及本目录导航 | 普通 Git clone |
| 私有 [zsure27/VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives) | 恢复目录、完整原始归档和逐项核验材料 | 有权限的账号获取目录；大文件按需下载 Release 附件 |

原始观测/动作、视频、参数、完整 tar 和 Git bundle 使用私有库的 [backup-archive-20261017 Release](https://github.com/zsure27/VLA-Quant-Archives/releases/tag/backup-archive-20261017)。公开代码库的普通 clone 不下载这些附件。操作步骤见[按需恢复方法](RESTORE_CN.md)，存储约定见[备份策略](../../docs/GITHUB_BACKUP_POLICY_20261017_CN.md)。

## 进行中的迁移 · 2026-10-19 检查点

| 项目 | 已核对的进度 |
|---|---|
| 完整对象远端下载 SHA 核验 | 89 个，共 15,963,223,602 字节 |
| 本地原文件清理 | 24,346 个，按文件大小合计 36,852,151,982 字节 |
| 尚在传输 | 4 个大归档；全量迁移和最终清理尚未完成 |
| 已回读核验的私有目录提交 | [`d6f6f66928d5636059198440b1dc474cfe35f533`](https://github.com/zsure27/VLA-Quant-Archives/tree/d6f6f66928d5636059198440b1dc474cfe35f533/catalog) |

清理大小为文件逻辑字节合计，最终实际释放空间另按磁盘实测记录。2026-10-17 的初始约 55.14GB 审计是当时两个备份根目录的范围；后续额外旧备份和 B5 历史片段单独登记。上述检查点不代表剩余文件已上传或已可删除。[机器可读审计摘要](LOCAL_AUDIT_20261017.json)保留初始范围和当前进度。

## 私有目录中的恢复入口

- `catalog/20261017/remote-verification.json`：仅纳入完整分块、下载 SHA 和整包恢复关系均通过的对象；用它运行恢复脚本。
- `catalog/20261017/original-archives.json`：最初 54 个归档的原路径、重复别名和完整 SHA。
- `catalog/20261017/extra-backup-manifest-20261018.json`：额外旧备份的路径映射。
- `catalog/20261017/coverage/`：展开文件与归档成员之间的恢复证明。
- `catalog/20261018/`：已发布的分批清理记录；后续检查点继续补充。
- `catalog/20261017/range-staging-metadata-20261018.json`：B5 历史下载片段的原始区间；这些片段缺少前缀，不能作为完整 B5 归档恢复。

只有某个对象的全部恢复门禁通过、对应目录已发布核验后，才按精确路径清理它覆盖的本地文件。活动上传源和唯一未核验数据继续保留。此前直接公开上传完整内部归档被自动审批拒绝的历史事件保留；用户已明确选择上述私有目的地。
