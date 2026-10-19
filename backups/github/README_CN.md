# GitHub 归档与恢复索引

| 仓库 | 保存内容 | 获取方式 |
|---|---|---|
| 公开 [zsure27/VLA-Quant](https://github.com/zsure27/VLA-Quant) | 代码、配置、分析、小结果及本目录导航 | 普通 Git clone |
| 私有 [zsure27/VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives) | 恢复目录、原始归档和逐项核验材料 | 有权限的账号获取目录；大文件按需下载 Release 附件 |

原始观测和动作、视频、参数、完整归档和 Git bundle 使用私有库的 [backup-archive-20261017 Release](https://github.com/zsure27/VLA-Quant-Archives/releases/tag/backup-archive-20261017)。公开代码库的普通 clone 不下载这些附件。操作步骤见[按需恢复方法](RESTORE_CN.md)，存储约定见[备份策略](../../docs/GITHUB_BACKUP_POLICY_20261017_CN.md)。

## 本次迁移完成 · 2026-10-19

| 项目 | 已核验状态 |
|---|---|
| 已审计迁移范围 | 原始附件和迁移审计证据已通过逐块下载 SHA 及完整对象恢复关系核验 |
| 本地原始备份副本 | 已按内容覆盖证明和逐文件回执完成授权范围的精确清理 |
| 当前保留 | 工作代码、小结果、恢复入口及必要的工作材料 |
| 详细记录 | [完整恢复清单](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261017/remote-verification.json)、[原始备份副本清理摘要](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261019/cleanup-final-summary.json) |

上述完成范围不包括未纳入本次审计的研究文件。审计缓存清理与旧布局附件整理也已分别核验完成；[最终完成摘要](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261019/migration-completion.json)保存对应回执。清理大小按逻辑文件字节与磁盘实际空间分别记录，详情保存在私有摘要中。[机器可读审计摘要](LOCAL_AUDIT_20261017.json)保留初始范围和历史检查点。

## 从私有目录恢复

- 先取得私有库当前恢复清单，按来源路径与完整 SHA 选择材料；不同对象的同名归档不能互相替代。
- 原始归档与其重复别名、补充材料、展开文件成员覆盖证明及迁移清理回执均在私有目录中保存。
- 使用[恢复步骤](https://github.com/zsure27/VLA-Quant-Archives/blob/main/docs/RESTORE_CN.md)或本目录的[操作说明](RESTORE_CN.md)，只下载当前需要的对象。
- B5只保存了历史下载的部分区间，完整包前缀缺失；这些片段不能视为完整B5备份。

只有内容的完整恢复门禁通过、对应目录已发布核验后，才能按精确路径清理其覆盖的本地副本。未核验唯一数据与工作文件继续保留。此前直接公开上传完整内部归档被自动审批拒绝的历史事件保留；用户已明确选择上述私有目的地，私有资产不得擅自公开。

[具体迁移文件及流量说明](https://github.com/zsure27/VLA-Quant-Archives/blob/main/docs/MIGRATION_CONTENT_AND_TRAFFIC_CN.md)保存实验分组、具体文件和完整读回校验的流量预算。
