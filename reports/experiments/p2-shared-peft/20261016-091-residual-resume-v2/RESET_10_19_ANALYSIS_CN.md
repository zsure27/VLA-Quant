# M0 reset10–19与累计150分析（2026-10-17）

三阶段全部exit0/revision11，PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL。实际argv/唯一输出/运行时锁、task/reset/派生seed/首个策略可见观测、每个NPZ SHA/动作/一次本体归一化/finite、真实首query零分支与PTQ/冻结/保存重载、100+100原始视频均通过，无程序异常。旧336科学源码和A4/S0身份未改。

## 结果及不确定性

本片PTQ15/100、S0仅初始化43/100；31救回3新增失败，净+28pp，任务经验聚类95%[13,44]pp，McNemar p=7.66013e-7。8任务净改善、0任务净变差（任务内仍存在break）。本片新增失败为task0/reset13、task3/reset10与reset12。

累计正式reset5–19各150：PTQ22（14.67%）、S062（41.33%）。44救回4新增失败，净+40/150=26.6667pp；任务经验聚类95%[11.3333,42.6667]pp、McNemar p=1.51383e-9。8任务净改善，其余2任务净0。统计仅同十任务开发，不是未见任务/盲测；微测不加分母，不混入旧046 Z/S数据。当前结论限于固定SVD初始补偿相对固定PTQ的恢复效果，不是新优化方向的创新证据，也没有证明普通LoRA普遍失效。

## 成本与失败限制

PTQ2584次策略查询/1783.878秒，S02116次/1586.120秒。墙钟含加载、checksum、首query门禁、轨迹和视频IO，查询数差异可受回合长度影响，不是模型纯延迟或INT2部署加速。S0无新优化步、无新教师标签，但历史额外80校准轨迹及SVD构造成本单列，不当作零信息预算。适配器19988480参数/39976960 BF16 tensor字节。

完整逐键成败、所有rescue/break、任务差值、累计和本片全部失败在同名results/reset-10-19-gate.json及cumulative-150-analysis.json。不能从标签推断夹爪/碰撞等物理失败机制。本片全部失败键如下：

ptq: task0/reset10, task0/reset11, task0/reset12, task0/reset14, task0/reset15, task0/reset16, task0/reset17, task0/reset18, task0/reset19, task1/reset10, task1/reset11, task1/reset12, task1/reset13, task1/reset14, task1/reset15, task1/reset16, task1/reset17, task1/reset18, task1/reset19, task2/reset10, task2/reset12, task2/reset14, task2/reset16, task2/reset17, task2/reset18, task3/reset14, task3/reset15, task3/reset16, task3/reset19, task4/reset10, task4/reset11, task4/reset12, task4/reset13, task4/reset14, task4/reset15, task4/reset16, task4/reset17, task4/reset18, task4/reset19, task5/reset10, task5/reset12, task5/reset13, task5/reset14, task5/reset15, task5/reset16, task5/reset17, task5/reset18, task5/reset19, task6/reset10, task6/reset11, task6/reset12, task6/reset13, task6/reset14, task6/reset15, task6/reset16, task6/reset17, task6/reset18, task6/reset19, task7/reset10, task7/reset11, task7/reset12, task7/reset13, task7/reset14, task7/reset15, task7/reset16, task7/reset17, task7/reset18, task7/reset19, task8/reset10, task8/reset12, task8/reset13, task8/reset16, task8/reset17, task8/reset18, task8/reset19, task9/reset10, task9/reset11, task9/reset12, task9/reset13, task9/reset14, task9/reset15, task9/reset16, task9/reset17, task9/reset18, task9/reset19

svd-init-only: task0/reset10, task0/reset13, task1/reset10, task1/reset11, task1/reset12, task1/reset13, task1/reset14, task1/reset15, task1/reset16, task1/reset17, task1/reset18, task1/reset19, task3/reset10, task3/reset12, task4/reset10, task4/reset11, task4/reset14, task4/reset17, task4/reset18, task4/reset19, task5/reset10, task5/reset12, task5/reset14, task5/reset15, task5/reset16, task5/reset17, task5/reset18, task5/reset19, task6/reset10, task6/reset11, task6/reset12, task6/reset13, task6/reset14, task6/reset15, task6/reset16, task6/reset17, task6/reset18, task6/reset19, task7/reset10, task7/reset11, task7/reset13, task7/reset15, task8/reset10, task8/reset12, task8/reset13, task8/reset16, task8/reset17, task8/reset18, task9/reset10, task9/reset11, task9/reset12, task9/reset13, task9/reset14, task9/reset15, task9/reset17, task9/reset18, task9/reset19

## 接续和备份

严格按原预注册顺序接reset20–29各100，再30–39/40–49，总各450，不按成绩方向改变条件。注册前重读当前5小时primary和盘，旧gate及本报告SHA纳入新锁。完成片归档服务器backups/091-residual-reset-10-19-complete-20261017-v1、本机D:/VLA-Backups/20261016-091-residual-resume-v2/同名，取得逐成员SHA后才认定双份；活动20–29排除。当前本报告供下一片锁定，后加归档说明须单独补充，不覆盖服务器锁定文件。
