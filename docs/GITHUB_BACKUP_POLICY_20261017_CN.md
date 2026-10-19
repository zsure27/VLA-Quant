# GitHub优先备份与本地清理策略

用户于2026-10-17要求后续尽量将实验备份放GitHub，减少本地磁盘占用。本策略覆盖此前必须长期保存全量本机物理副本的存储选择；原始科学数据、身份、SHA和可恢复性保持。

## 存储分工

|材料|位置|本地保留|
|---|---|---|
|代码、配置、报告、小指标、SHA及恢复索引|普通Git|工作必需文件|
|视频、原始NPZ、参数、完整tar/bundle|私有归档库zsure27/VLA-Quant-Archives的Release附件；超限分块|通过远端恢复核验后按需缓存|
|凭据、私钥、真实token|不进入归档仓库|用户原安全位置|

用户已明确选择私有GitHub归档仓库，现已创建并核验[zsure27/VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives)。公开代码仓库zsure27/VLA-Quant继续保存代码、分析、小结果和恢复索引；完整原始归档上传至私有库的[backup-archive-20261017 Release](https://github.com/zsure27/VLA-Quant-Archives/releases/tag/backup-archive-20261017)，需要有权限的账号恢复。不得擅自将私有资产改为公开。

此前自动审批拒绝将完整内部实验归档直接公开发布，保留为历史事件。用户选择已通过私有归档目的地解决该阻断。本次已审计范围的私有迁移及已授权原始备份副本清理已完成；完整恢复目录、下载核验和清理回执见[云端索引](../backups/github/README_CN.md)。B5只有历史部分区间，不能称为完整备份。审计缓存与旧布局附件也已分别核验整理，回执见[私有完成摘要](https://github.com/zsure27/VLA-Quant-Archives/blob/main/catalog/20261019/migration-completion.json)。

GitHub普通Git阻止超过100MiB文件。Release单附件须小于2GiB；本项目保留既有1GiB布局，并对独立续传使用256MiB布局；按恢复清单中的块序、offset和完整SHA恢复，布局回执相互隔离，普通clone不携带这些大附件。官方说明：[large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。不默认引入付费LFS。

## 清理门禁

1. 审计所有候选的绝对路径、大小、SHA和重复关系，跳过reparse/junction；区分唯一数据和重复展开payload。
2. 扫描真实凭据、私钥和token；冻结归档不能直接改字节，若含秘密需独立生成安全归档并保留科学恢复映射。
3. 以内容SHA命名附件，记录原文件整SHA、分块顺序/offset/大小/SHA、asset ID/URL和来源。相同内容只上传一次。
4. 验证远端uploaded状态、长度和digest；完整流式下载每块验证SHA，不在本机再生成大副本。核对分块连续offset、大小、顺序与整包SHA。未完成对象继续保留原件。
5. 将恢复清单和成员证明发布到私有Git，回读核验提交中路径所指blob及文件SHA。展开payload只有其每个文件内容均被该已核验归档覆盖时可删；小sidecar及恢复映射先保存。按完整归档逐项放行，保存清理前后清单与实际释放空间，逻辑字节合计和磁盘实测分别报告。
6. 删除使用PowerShell原生精确路径，核对留在本任务批准目录内；当前工作代码、未上传唯一数据和凭据保持。

## 后续实验

每个阶段完成后保存小分析/manifest到Git，完整原始材料上传已选私有归档库。阶段包与最终集合通过索引复用，同一归档使用唯一上传计划和可续传回执，避免重复传输和反复复制。结束前完成云端恢复验证；后续只按[恢复说明](../backups/github/RESTORE_CN.md)下载所需对象。GitHub可用性或权限阻断时保存现有数据并报告。

## 迁移内容与流量预算

具体文件清单及预算保存在[私有归档说明](https://github.com/zsure27/VLA-Quant-Archives/blob/main/docs/MIGRATION_CONTENT_AND_TRAFFIC_CN.md)。开始大批量传输前应明确文件总量、上传与完整读回校验的最低流量，以及失败重试预算。

大文件可用独立256MiB分块工具`github_archive_store_v3.py`，默认16路；不同分块布局使用隔离回执。不可恢复传输失败会阻止排队上传，活动上传在下一次进度报告处中断。已完整校验的对象可直接复用，本地源文件保留至恢复门禁通过。
