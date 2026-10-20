# M0首50精确恢复分析（2026-10-17）

R0本次120真实短回合/重复性通过，与上轮全部plan/sources/normalization/within/cross/runs指纹内容相同。原PTQ50与S0完整47的NPZ/动作/一次归一化/实际seed/首观测/视频全过，仅补task9/reset7–9三回合；真实S0零分支等于PTQ在task9/reset7对应输入的原始动作逐位相同，保存重载与冻结契约通过。新三阶段全部exit0，最终PASS_B3_INIT_ONLY_SEGMENTED_PAIRED_PROTOCOL。原S0没有exit0的事实保留，不伪造为旧完整进程；旧未完成13查询排除正式统计，但保留原始证据。

## 配对结果

PTQ7/50（14%），S0仅初始化19/50（38%）。13救回、1新增失败，净+12/50=24pp；任务经验聚类95%[6,42]pp，McNemar p=.0018310546875。6任务净改善、1任务净变差。新增失败仅task8/reset6；3个补回合task9/reset7–9均为共同失败，不依据此方向选择续片。

这支持冻结SVD初始补偿具有恢复作用，但仅同十任务开发50回合、单配方，不能替代1000步训练对照或证明新补偿方向创新。原046 S1000首50/450只历史，未在本文跨分母相减；需单独实际初始观测/协议桥接后才做同键参照。不读取holdout标签。

## 成本与全部失败

PTQ1292次查询，S0完整回合1103次查询，旧中断13次查询不计正式但作为浪费成本保存。旧S0尝试873.997秒含中断，新恢复181.017秒，含加载/校验/真实首query门禁/视频/IO，不是纯推理延迟或INT2 kernel收益。S0仍19988480参数、39976960 BF16 tensor字节，使用历史独立80轨迹校准/SVD成本并非零成本；本轮无优化步或新教师标签。

全部失败及逐键配对结果保存在同名results/first50-recovered-gate.json，下面附完整失败键，不能仅从失败标签推断夹爪/碰撞等机制。

ptq: task0/reset5, task0/reset6, task0/reset7, task0/reset8, task0/reset9, task1/reset5, task1/reset6, task1/reset7, task1/reset8, task1/reset9, task2/reset5, task2/reset7, task2/reset9, task3/reset5, task3/reset9, task4/reset5, task4/reset6, task4/reset7, task4/reset8, task4/reset9, task5/reset5, task5/reset6, task5/reset7, task5/reset8, task5/reset9, task6/reset5, task6/reset6, task6/reset7, task6/reset8, task6/reset9, task7/reset5, task7/reset6, task7/reset7, task7/reset8, task7/reset9, task8/reset5, task8/reset7, task8/reset8, task9/reset5, task9/reset6, task9/reset7, task9/reset8, task9/reset9

svd-init-only: task0/reset5, task0/reset9, task1/reset5, task1/reset6, task1/reset7, task1/reset8, task1/reset9, task4/reset6, task4/reset7, task4/reset8, task4/reset9, task5/reset5, task5/reset6, task5/reset7, task5/reset8, task5/reset9, task6/reset5, task6/reset7, task6/reset8, task6/reset9, task7/reset7, task7/reset9, task8/reset5, task8/reset6, task8/reset7, task8/reset8, task9/reset5, task9/reset6, task9/reset7, task9/reset8, task9/reset9

## 下一片

按原预注册顺序接reset10–19各100，随后20–29/30–39/40–49，总各450；只改offset/count/output/lock。续跑取决于协议、实时primary额度与磁盘，不按成绩方向。首50完整恢复备份在新服务器包及D:/VLA-Backups/20261016-091-residual-resume-v2，取得逐文件核验才称双份。原数据和原归档不可变。

## 完整双份归档完成

服务器/backups/091-residual-recovered-first50-complete-20261017-v1，本机D:/VLA-Backups/20261016-091-residual-resume-v2/同名；424158612字节SHA30c48668178c8fb209e140c86c3e7643be9d13f77ca451f376ee3716155eb544，2575文件/2576解压成员、100原始视频逐项SHA通过。包括原S0未完成残段作为证据但不计正式样本；活动reset10–19全部排除。原始tar不随此补充分析改动，补充回执以同名results/Git和控制根保存。
