# 2026-10-17 本地备份与仓库整理

## 初始审计（历史范围）

2026-10-17初始审计的本地完整备份约55.14GB：C盘repo/.local/archives 35,558,761,029字节，D:/VLA-Backups 19,585,039,125字节。大量是完整阶段tar在单阶段目录、15%集合和最终集合中的重复副本，以及归档后的展开payload。重复量仍按独立SHA与文件覆盖证明核对，不提前声称释放空间。约10.24GB旧SVD输入tar为本机唯一副本之一，未云端核验前保留。

代码仓库zsure27/VLA-Quant公开，完整数据归档改用用户选定并已核验的私有仓库zsure27/VLA-Quant-Archives。普通Git远端约50MiB，本地追踪树243MiB，最大blob14.2MB。主问题是本地完整数据与重复归档，不需要改写Git历史或删除科学证据。本次整理主要减少本地磁盘占用。

## 已整理的仓库入口

- 首页更新为10月17日三轮实验、小规模探索和GitHub备份索引；旧README原字节保存在docs/archive，SHA由索引记录。
- reports/scripts/P2/P5索引取消过期活动状态，保留历史路径；不搬动冻结实验目录、源码、lock与原始结果。
- AGENTS原52KB历史原字节独立保存，现行文件只列当前规则和入口。历史相对路径按原工作目录解释，不改写其字节。
- backups/results改为GitHub大附件优先、小索引入Git、本地按需下载。.gitignore新增压缩包/视频/bundle/下载分块及临时缓存保护。
- 两份跨轮综述通过明确document role登记，无需伪造新实验results；校验器仅允许声明过且有原报告来源的summary例外。

## 上传和删除状态

此前自动审批拒绝把完整实验内部归档直接公开发布，原因为概括备份授权未明确具体公开载荷。用户随后明确选择私有归档仓库，现已创建并核验[zsure27/VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives)，原公开上传阻断保留为历史记录。

本次已审计材料已迁移至私有库的[backup-archive-20261017 Release](https://github.com/zsure27/VLA-Quant-Archives/releases/tag/backup-archive-20261017)，通过逐块流式下载SHA及完整对象恢复关系核验后，已按精确内容覆盖证明和逐文件回执清理授权范围的原始备份副本。[完整恢复清单](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261017/remote-verification.json)和[清理摘要](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261019/cleanup-final-summary.json)保存在私有库；公开入口见[云端索引](../backups/github/README_CN.md)。工作文件、小结果、凭据和恢复入口保留；未纳入本次审计的研究文件不据此清理。B5只保存历史部分区间，不能恢复完整包。审计缓存清理和旧布局附件整理也已分别通过门禁，详见[私有完成摘要](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261019/migration-completion.json)。完整归档使用Release附件，分块布局以恢复清单为准，不放入普通Git或默认引入付费LFS。

## 核验

仓库布局验证通过：58报告会话、60结果会话、2明确跨轮综述。既有116份历史精确去重映射全部通过。本次已审计迁移范围已取得最终私有恢复目录发布门禁，并完成已授权原始备份副本的精确清理；逐项证据由私有清单和清理摘要保存。历史初始审计和布局核验范围保持，不能扩大为全部研究文件或完整B5材料均已备份。
