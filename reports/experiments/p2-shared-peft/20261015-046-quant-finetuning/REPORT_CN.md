> 2026-10-15最新覆盖：reset10–19已完成全部exit0/配对门禁，本片Z0/S95（各100），累计150为Z1/S144，143救回0新增失败、任务95%[92,98.667]pp。已按协议/资源接reset20–29各100，09:30UTC实测RUNNING revision3/Z侧，supervisor21885/GPU21887仅历史。唯一活动计划B3-continuation-v1/reset-20-29/plan.json，SHA7ba43a6a179e9d0cf6bd02983884b725ccbc5927020b7d696a6da54c1038cc3c。详见RESET_10_19_ANALYSIS_CN.md。

> 最新覆盖（2026-10-15）：首50计划12阶段全部结束，Z=1/50、S=49/50，48 rescue/0 break，任务经验95%区间[90,100]pp，配对硬门禁通过。按既定顺序已启动reset10–19每侧100；当前唯一活动计划为控制根B3-continuation-v1/reset-10-19/plan.json，文件SHA8767cb6907d99142aa64910b00282e3de7d02d2499969131433edba84b1fedc9；status为同目录control/20261015-046-quant-finetuning-v1-reset-10-19/status.json。08:52 UTC实测revision6、S侧已接续，supervisor11229、wrapper16378、策略进程16379，磁盘12,359,987,200字节；PID及阶段只代表当时历史。源/已训练adapter不变，不重跑首50。详细解释见FIRST50_ANALYSIS_CN.md。

# 046量化微调对照：启动与门禁回执

日期：2026-10-15。状态为**真实GPU计划已启动，训练与闭环结果待完成**；不能将启动或CPU PASS当作新增成功率。

> 最新GPU门禁更新：两侧10步烟雾及`PASS_B3_INITIALIZATION_SMOKE_PAIR`已完成。真实零输出/保存重载/冻结参数、两侧相同目标白名单/参数数目/教师/训练序列均通过；每侧19,988,480参数。最新观测revision12为`train-1000-standard-zero`，执行器已自动进入正式训练；尚无新闭环成功率。门禁回执已同步同名results目录。

> 后续覆盖：两侧正式1000步及`PASS_B3_INITIALIZATION_TRAIN_PAIR`均已完成；micro协议通过，Z=0/10、S=10/10（与正式重叠，不加总）。当前首50 Z侧正常运行revision30。训练完整增量双份756,868,488字节、SHA d0956ebefb3adb03d5b4b1cb057c5281e208b54a4b946b4cd109f1dd9121bf39、664项本机逐文件核验通过；完整活动评测未纳入；另含35个已完成视频，见范围补充。详见同目录`TRAINING_ANALYSIS_CN.md`，不能将微测当最终450结论或复用历史B3成绩替代新S。

## 已完成

- 用户开启046并明确授权按10-13研究计划实验。SSH端口38230，实际hostname `autodl-container-90824aab47-6dc00ae3`；ED25519公钥与已有可信记录匹配。本轮使用独立known_hosts。
- 实测4090 24564MiB、cgroup内存120GiB/16CPU。起跑持久盘空闲14,959,501,312字节（13.93GiB）。`free`显示的宿主机总内存不是容器配额。
- 收尾工具已同步到持久盘，SHA/编译通过：backup工具SHA `698f0b210d4e0ecd5cecdea83122f6358fcf45468615b967010f0bda973a68a4`，shutdown工具SHA `efed5fdf5e7a98c900306f9ceba823f4c2b417ae36194ec6ebe59fd3f17ff12c`。
- 091完整会话归档主清单22,244项、数据清单15,389项逐文件SHA全过；完整7B checkpoint与profile身份一致；B3训练80观测、独立SVD校准80轨迹及数据角色/交集审计通过。原7B checkpoint仍为服务器数据，不把实验归档说成完整本机checkpoint副本。
- 真实评测器调用顺序的无模型观测探针：10任务×2初态×3协议×2重复，共120短回合。三个协议内部重复均完全一致，`PASS_CPU_PROTOCOL_OBSERVATION_BRIDGE`，818.08秒。这不代表三个协议彼此相同，也不提供模型闭环收益证据。
- 独立source-v2包含358项内容SHA；3项CPU契约验证标准零输出的确定性/随机流隔离、非零学习梯度、冻结底座与保存重载。历史SVD初始因子和共享教师cache逐文件SHA及原始实现/数据身份均通过。
- 固定仓库`zsure27/VLA-Quant`预注册提交`fc3e451d4c42de0b44fb7c36562531079e651eed`已非交互推送并核验远端main一致。

## 保留的前置失败

R0-v1在环境初始化因空EGL设备值失败（`invalid literal for int() with base 10: ''`），没有进入策略评测。保留失败回执；v2显式`MUJOCO_EGL_DEVICE_ID=0`，科学参数不变，通过完整原定探针。CPU测试第一次用模块名寻找`tests.test_b3_initialization_v1`失败，未执行测试；改为标准unittest discover后相同源码3项通过。最后部署补强了成对审计的实际派生seed和程序错误检查，独立source-v2再次通过CPU契约。所有旧版本保留。

## 首50原始GPU计划（已完成）

- 源码：`/root/autodl-tmp/qvla-repro/control/20261015-046-quant-finetuning-v1/source-v2`；tar SHA `7e46f0403260a0c0297496c57050af9a11f75a7d9cdadb82f686a376c7594c1d`。
- 计划：同控制根`B3-plans-v1/first50-plan.json`，文件SHA `f6d2bf0a19cd6b6054f9ee6c4f77e18a07c99802624971bd69974c6ef5be8992`；canonical计划SHA `480acaf70e2a7cea2f4aabef755d3e06b4162d2be6859f7330634f64f688199f`。
- 状态：同控制根`B3-plans-v1/control/20261015-046-quant-finetuning-v1-first50/status.json`。
- 产物：`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261015-046-quant-finetuning-v1`。
- 北京时间15:14观测：`RUNNING`，revision3，`train-10-standard-zero`；supervisor2156、wrapper2157、GPU进程2158；GPU显存16,295MiB，持久盘空闲14,871,298,048字节。这些均为当时观测，后续检查实际进程，不沿用旧PID判断存活。

12阶段自动执行：Z/S各10步烟雾→训练可比性gate→Z/S各1000步→训练可比性gate→Z/S各10回合micro→配对协议gate→Z/S各50正式回合→首片硬gate。任一阶段失败停止后续，保留所有输出。两侧同A4、224语言Linear/rank8、19,988,480参数、80观测/教师、seed7和1000步；正式训练从初始因子重新开始，不接着烟雾训练累计1010步。S侧复用训练前因子，重新训练，不将历史已训练B3冒充新S结果；SVD历史采集/分解/IO成本单列。保存0/100/250/500/1000步checkpoint供预定离线诊断，不能用正式开发成败选checkpoint。

## 接续与成本限制

目前不可变计划只放行每侧首50；整体目标为同任务开发reset5–49各450。首50包含在450，micro不加总。首片后先分析rescue/break、任务区间/McNemar、失败和实际成本，再核实时额度/磁盘并预注册后片，不按成功率方向决定续跑。当前只比较Z/S；历史BF16/A0/A4参照须在协议与实际输入绑定通过后才能并入同条件比较，不直接拼接旧分母。

先完成基础初始化对照，再分别推进时间覆盖、动作目标/执行敏感残差方向及C-SQ-v1等价/梯度/恢复；这些后续项仍受各自门禁约束，当前没有宣称已全部实现或放行。暂停扩展routing，不触碰offline_final_holdout标签。

本轮046心跳15分钟ACTIVE，阶段未变静默；完成/失败/空闲待接续/额度门限时处理。只看实时五小时primary：约15%停扩展并完整分析双份归档、固定GitHub SHA；约10%停新GPU并原生关当前046，留至少3%。运行保留6GiB。当前不关闭Windows，不继承091本机关机许可。

## 首50完整评测双份增量

服务器`/root/autodl-tmp/qvla-repro/backups/046-b3-initialization-first50-complete-20261015-v1`；本机repo `.local/archives/20261015-046-quant-finetuning/`下同名物理目录。`complete.tar.gz` 399,964,959字节，SHA256 `9e3b38020e743c083c6712a72cdaf8fd48e6c9134b187fc1802c429c0ebd1952`。服务器压缩成员与本机2995项SHA/2996成员全部核验通过，含micro与first50完整轨迹/观测动作/命令/退出码及120原始视频（每个视频episode ordinal和成败与gate逐项一致）。实际冻结源码仅按358项清单逐文件纳入；当前reset10–19不包含，后续仍需完成片增量。与训练增量和091来源归档共同保存当前完成材料，不声称本机有完整7B checkpoint。

训练包范围已附SCOPE_SUPPLEMENT_V2.json：原训练包意外包含35个已完成视频，全部原SHA有效，压缩包保持不变，不将它当成完整评测归档。源码归档工具已改为清单白名单，未改变活动GPU冻结源码。

## 完成分片的双份归档回执

reset10–19完整增量服务器`/root/autodl-tmp/qvla-repro/backups/046-b3-reset-10-19-complete-20261015-v1`，本机repo `.local/archives/20261015-046-quant-finetuning/`下同名目录。679,500,413字节，SHA256 `2cec7fcaf7059a61b18c8d25a6df97947801a61524fdc929b76a264a97e0da56`；4777项SHA/4778成员和200个原始视频逐项核验通过，保存全部轨迹/实际命令/退出码/门禁/358源码/150配对分析。活动reset20–29不包含。CN报告和最新本机回执另同步控制根，与冻结归档分开保存，不改原tar。

训练、micro/首50和本片的完整服务器/本机索引见同名results/COMPLETED_STAGE_BACKUP_INDEX_V1.json；原091来源归档及7B checkpoint保持原角色。阶段增量不是最终原生关机许可回执，最终仍须完整会话归档及服务器root短HEAD bundle。
