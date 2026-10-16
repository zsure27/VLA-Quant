# B1时间覆盖：独立闭环准入与首50卡

已完成Early80/Phase80各10步和独立1000步、配对训练gate通过；不根据两侧不同观测的训练loss选择checkpoint，两侧均固定1000。每侧19988480参数，同A4/SVD初始化/序列/优化器/损失，唯一变量取帧时间。

新评测版本b1-temporal80-eval-v1-20261017，模型B1-temporal-Early80-v1与B1-temporal-Phase80-v1。新增artifact loader逐项核验训练version/variant/真实1000步/paired gate/448因子形状/源SHA/初始与教师材料/冻结/保存重载/训练角色，拒绝把10步或旧B3/S0重命名；旧336源和旧validators不改。沿用原真实LIBERO完整调用顺序/paired派生seed/动作后处理/视频，R0当前重启内容桥接已过，不把新入口当独立环境复现。

每侧第一真实query：原始双相机、本体和指令须与M0已完成PTQ的对应task0/reset5原始NPZ逐项一致；本体只归一化一次，临时把LoRA B置0得到的真实动作须逐位等于该PTQ已执行动作；恢复训练后参数、保存重载、冻结底座/头/本体及RNG恢复全部通过。仅在独立raw副本上探针，实际首次query不重复归一化。任一失败停止，不放宽。

注册六阶段：micro Early10→Phase10→micro严格配对gate→first50 Early→Phase→first50严格硬gate。GPU新启动需CPU元数据与实际状态shape契约/负例通过、预注册Git远端核验、GPU闲、盘12GiB/运行留6GiB。每阶段独立输出/lock，失败自动停止。首50通过后保存全部救回/新增失败/每任务区间/McNemar/全部失败/成本和完整双份归档；然后按原研究计划协议资源顺序注册后续100片直至450/侧，不按成绩方向选择。micro重叠不加分母，不混M0/046标签，不读holdout。首50前无闭环收益结论。

模型原始视频与完整轨迹/命令/参数/来源/报告均保存服务器和本机D盘逐SHA；GitHub固定zsure27/VLA-Quant只存代码/规划/小结果。04:52:52实时primary API已经确认自然刷新1792183963→1792201971，首次窗口豁免结束，恢复15%备份/10%关091/留3%；最终完整核验/原生execute=true/PAUSED/总结后正常关本机。
