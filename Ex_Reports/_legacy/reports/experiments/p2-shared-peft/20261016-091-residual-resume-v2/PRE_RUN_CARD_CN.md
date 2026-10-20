# 091恢复运行卡（2026-10-16）

用户已重新开启091端口18575，授权按既定两方向计划继续；本轮结束关闭本地电脑。该授权只适用于本次恢复运行，覆盖此前本轮不关Windows。先完整双份归档/分析/Git远端SHA、091原生execute=true、心跳PAUSED、发送总结，最后请求Windows正常关机；不强制关闭应用，关键步骤失败不关本机。

## 真实准入

hostname与既有可信ED25519通过，冻结原source-v1共336文件逐SHA相同。当前4090空闲，盘18.43GB；实时primary剩98%、reset1792183963实际新窗口，仅当次历史。新候选12GiB/后片8GiB/运行6GiB，15%备份/10%停GPU并原生关091/留3%，不新增豁免。C盘仅约2.1GB，后续大备份写D:/VLA-Backups/20261016-091-residual-resume-v2；旧唯一数据保持。

## 精确恢复边界

原PTQ首50 exit0完整50；原S0首50有48次开始、47次完整结束，第48回合task9/reset7中断。已完成部分为task0–8各reset5–9、task9/reset5–6。缺失只task9/reset7–9；原47回合须逐事件/实际manifest、查询NPZ SHA/动作/归一化/finite/视频复核后复用，未完成task9/reset7旧残段保留排除。禁止覆盖原输出、伪造exit0或重复计算分母。

先执行本次R0真实评测调用顺序观测重复性。恢复适配层仅限定原run_task调用的task/reset调度，不改科学模型、原336文件、量化、初始化、教师、动作后处理或每回合seed顺序。先CPU调度/覆盖/缺失负例契约，再真实首查询与PTQ对应task9/reset7数值桥接，任一失败停止并诊断。完成后新分段审计应精确50键，比较原PTQ首50对应观测/seed；跨片完整结束证据与新进程退出码分别记录，不能伪造旧S0 wrapper exit0。

首50门禁通过后先保存rescue/break、任务区间/McNemar、全部失败与成本；再按原计划reset10–19、20–29、30–39、40–49每侧100顺序注册唯一计划，总各450。按协议资源额度接续，不按成绩方向延长。旧Z/S450仅历史参照；micro重叠不加。后续仍先M0机制控制，再Early80/Phase80、动作敏感方向、独立动作目标及C-SQ门禁，不扩routing/不读holdout标签。

## 已实际接续

R0/重启观测内容桥接及47+3首50严格审计均通过，见FIRST50_RECOVERED_ANALYSIS_CN.md。当前reset10–19唯一plan SHA517c5a9678262b01310aac8266130735236a76c983029fccf9b2ee85771c6501，00:19:40启动supervisor2863，实测RUNNING revision3/PTQ、GPU2865/显存16621MiB（仅检查时历史，后续读实际）。服务器自动PTQ→S0→硬gate，不重跑旧first50。活跃15分钟心跳vla-091-vla已ACTIVE。后续大备份D盘，旧C盘唯一数据保持。

本轮独立Windows收尾脚本scripts/shutdown_local_091_residual_resume_v2.ps1已解析通过、尚未执行；需要本轮授权SHA/实验终止/GPU闲/双份SHA/分析/Git远端核验/匹配091原生execute=true/心跳PAUSED/已发送完整总结均具备后，才允许-Execute请求shutdown.exe /s /t 0，不强制关闭应用。
