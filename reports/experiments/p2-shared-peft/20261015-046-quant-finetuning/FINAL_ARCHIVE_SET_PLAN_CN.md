# 450结束后的完整会话归档集合与原生关机门禁

这是待执行准备方案，不是已完成收尾回执。必须等40–49硬门禁和450严格汇总完成，并确认GPU空闲后执行最终封存；新科研项仍按研究计划各自门禁和实时额度/磁盘判定。

## 避免重复压缩全部阶段

最终完整会话可以是自包含归档集合目录，不要求再次把六个已完整验证的大tar包套一层gzip。服务器建议新建`/root/autodl-tmp/qvla-repro/backups/046-b3-initialization-final-20261016-v1`，本机使用既有物理`.local/archives/20261015-046-quant-finetuning/`同名目录。将训练、micro+首50、10–19、20–29、30–39、40–49六个完整tar分别用硬链接放进最终目录的`stage-archives/`，绑定固定SHA和逐成员/本机回执。这些硬链接不是新增独立物理副本；最终目录完整包含各tar文件，传输/恢复时无需依赖外部相对路径。目录集合需同步本机（本机也可硬链接已核验tar避免重复空间）。不删除原始评测数据或已有tar。

## 补充封存及完整性

建`session-supplement.tar.gz`，包括最新完整reports/results中文分析与原始小结果、六包本机/服务器回执及范围补充、独立的完整会话索引、所有计划/runtime-lock/启动与完成status/监督器日志、控制根脚本/部署身份/R0失败与成功证据、以及阶段包未覆盖的当前会话文件。不要把正在生成的最终归档自包含，不反复递归收进旧大tar，也不要覆盖已硬链接归档的源文件。

读取六包MANIFEST.json和VIDEO_MANIFEST.json，建立原始source路径→SHA覆盖集合；遍历本轮session目录的全部实际文件（排除明确的缓存），未被阶段包覆盖或SHA不同的文件进入补充包。完成回合全部920个原始视频（正式900+micro20）需由视频清单逐文件覆盖和SHA核验；micro重叠成败不加到正式分母，但视频文件仍完整保留。冻结358源码已被每个评测包覆盖，需再次核对实际源码SHA。完整7B checkpoint仍为服务器原有全checksum，不声称本机有完整7B副本；091来源归档角色和SHA保留。

## Git和收尾工具要求

标准`closure-tools/vla_shutdown_remote.py`读取最终目录里的`SHA256SUMS.txt`与`RESULTS_SHA256SUMS.txt`，逐项验证相对路径在目录内和文件SHA；并要求`VLA-Quant-<服务器/root/VLA-Quant的git rev-parse --short HEAD>.bundle`真实存在。不能只保留完整HEAD名bundle。可用服务器/root/VLA-Quant生成该short名bundle；另把本机当前已经推送并远端SHA核验的研究代码/报告Git bundle一同保存，用明确文件名区分两套Git身份。冻结执行源码是source-v2归档树，不将它冒充有.git的工作区。

最终两份SHA清单应覆盖六个tar、补充包、Git bundle、完整会话索引/范围与回执等实际文件；results清单至少覆盖所有结果tar。核对服务器集合每项SHA及各tar成员；本机集合每项SHA及此前六包逐文件回执，补充包也解压逐文件核验；恢复路径、参数、数据、视频、代码/计划、分析和Git全部齐全后，才能使用该目录作final shutdown fallback。

调用标准shutdown工具不带--execute先做dry run，此时必须GPU空闲。正式执行前再次确认本轮实验终止、完整会话服务器/本机/Git远端SHA均过，原生回执execute=true并保存本机。阶段增量不能直接冒充最终shutdown gate。收尾后暂停本轮15分钟心跳；不声称SSH关闭等于平台OFF/停止计费独立验证。本轮无Windows关机授权。

## 本轮Windows关机授权更新

用户明确要求本轮运行结束后关闭本地电脑。本条覆盖本轮旧“不关Windows”，只限当前046完整运行，不继承未来。继续当前实验及实时15%/10%/3%门限。最终先核验全部实验终止/GPU空闲、服务器完整持久盘归档/本机完整副本/同名分析/GitHub远端SHA，再取得046原生关机执行回执，暂停本轮心跳，向用户发送完整结果与本机关机即将执行摘要，随后请求Windows原生关机。任何关键备份/GitHub/远端关机失败则保持本机运行，保留数据并继续处理或通知，不重复要求许可。当前只登记授权，尚未执行Windows关机。
