# GitHub优先备份与本地清理策略

用户于2026-10-17要求后续尽量将实验备份放GitHub，减少本地磁盘占用。本策略覆盖此前必须长期保存全量本机物理副本的存储选择；原始科学数据、身份、SHA和可恢复性保持。

## 存储分工

|材料|位置|本地保留|
|---|---|---|
|代码、配置、报告、小指标、SHA及恢复索引|普通Git|工作必需文件|
|视频、原始NPZ、参数、完整tar/bundle|私有归档库zsure27/VLA-Quant-Archives的Release附件；超限分块|通过远端恢复核验后按需缓存|
|凭据、私钥、真实token|不进入归档仓库|用户原安全位置|

用户已明确选择私有GitHub归档仓库，现已创建并核验[zsure27/VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives)。公开代码仓库zsure27/VLA-Quant继续保存代码、分析、小结果和恢复索引；完整原始归档上传至私有库的[backup-archive-20261017 Release](https://github.com/zsure27/VLA-Quant-Archives/releases/tag/backup-archive-20261017)，需要有权限的账号恢复。不得擅自将私有资产改为公开。

此前自动审批拒绝将完整内部实验归档直接公开发布，保留为历史事件。用户选择已通过私有归档目的地解决该阻断，当前上传和流式下载SHA核验正在进行，本地删除尚未执行。状态以backups/github索引及实际回执为准，不能仅凭仓库已创建或部分附件通过就认定全部备份完成。

GitHub普通Git阻止超过100MiB文件。Release单附件须小于2GiB；本项目默认按1GiB分块，避免普通clone携带大数据。官方说明：[large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。不默认引入付费LFS。

## 清理门禁

1. 审计所有候选的绝对路径、大小、SHA和重复关系，跳过reparse/junction；区分唯一数据和重复展开payload。
2. 扫描真实凭据、私钥和token；冻结归档不能直接改字节，若含秘密需独立生成安全归档并保留科学恢复映射。
3. 以内容SHA命名附件，记录原文件整SHA、分块顺序/offset/大小/SHA、asset ID/URL和来源。相同内容只上传一次。
4. 验证远端uploaded状态、长度和digest；完整流式下载每块验证SHA，不在本机再生成大副本。组合整SHA及tar成员清单支持复原。
5. 展开payload只有其每个文件内容被云端已验证归档覆盖时可删；小sidecar及恢复映射先存Git。保存清理前后清单与真实释放空间，不把硬链接逻辑大小当物理回收。
6. 删除使用PowerShell原生精确路径，核对留在本任务批准目录内；当前工作代码、未上传唯一数据和凭据保持。

## 后续实验

每个阶段完成后保存小分析/manifest到Git，完整原始材料上传已选归档库。阶段包与最终集合通过索引复用，不反复复制。结束前完成云端恢复验证；只按需要下载。GitHub可用性或权限阻断时保存现有数据并报告，不制造“已异地备份”的结论。
