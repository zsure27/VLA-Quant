# reset30–39完成门禁与累计350配对分析

2026-10-16，三阶段全部exit0、PASS_B3_INITIALIZATION_PAIRED_PROTOCOL，实际命令/派生seed/首观测/输入SHA/finite均过。本片Z=0/100、S=95/100；95救回/0新增失败，差95pp，任务聚类经验95%[89,100]pp，精确McNemar p5.04871e-29。

累计reset5–39，每侧350：Z2（0.5714%）、S334（95.4286%）；332救回/0新增失败，净94.8571pp，任务经验95%[91.7143,97.7143]pp，精确McNemar p2.28597e-100。micro与正式重叠，不加总；同十任务反复开发初态，不是未见任务/盲测，不读取holdout标签。

本片五次共同失败：task5/reset31、37、38，task6/reset39，task9/reset36；Z本片全部失败。累计S十六次失败均共同失败，完整清单见ANALYSIS.json。总成败不足以判定抓取/夹爪/搬运因果，完整动作/观测/视频保留。

这支持当前A4、80早期训练观测、1000步设置下SVD初始化的工程恢复作用；S另用80独立校准轨迹，不是等总前向/信息预算，不能外推所有标准零输出LoRA。两侧均真实更新，S新权重与历史B3非逐位一致，旧425/450不拼入本轮。

本片查询Z2800/S1400；失败执行更长，不能以查询/墙钟差称模型内核加速。两侧各19,988,480参数，BF16 tensor39,976,960字节、文件40,133,618字节，不含底座/激活/优化器与额外SVD教师成本；当前BF16承载伪量化，没有真实INT2内核或对应存储压缩证据。

北京时间00:14:51读取当前task API响应（00:14:44），primary剩87%、reset1792095433，仅历史。起跑磁盘8,958,566,400字节满足8GiB续片；按原顺序接最后40–49各100，不依据成绩方向，模型/adapter/源码不变，仅offset/count/output/lock变化。最后计划SHA01fc357c712a3a3965b89c286293acce18bbbd0f48786ee310d06a6d29785938，supervisor46423为00:16:21启动历史，后续读实际。

本片完整200视频增量仍须服务器/本机SHA核验，活动40–49排除；硬链接暂存不计独立物理副本，完整tar和核验的本机副本为便携归档。最终仍须完整会话索引/报告/Git/root短HEAD bundle核验。本轮不关Windows。

## 完整双份增量回执

30–39完整增量服务器`/root/autodl-tmp/qvla-repro/backups/046-b3-reset-30-39-complete-20261016-v1`，本机repo `.local/archives/20261015-046-quant-finetuning/`同名目录。685,081,029字节，SHA256 `06f3f20ebbf472acf9708d6964ae980c2da23bd7bec16fade37e0cd31cfd3aaa`；服务器压缩成员/本机4791项SHA、4792成员、200原始视频逐项核验通过，活动40–49排除。五包完成索引同名results/COMPLETED_STAGE_BACKUP_INDEX_V1.json。

最终归档集合的具体门禁和省空间封存步骤见FINAL_ARCHIVE_SET_PLAN_CN.md，尚未执行最终封存或关机。服务器root短HEAD bundle已准备并验证：closure-materials-v1/VLA-Quant-852d350.bundle，47,469,061字节、SHAe134858c069064667c1a79c05016ed99bbd7249cd06b66065ca4586dde211bd2。必须在最终目录纳入SHA，再额外保存当前研究Git bundle；不能拿这份旧root身份代替新执行源码或最新报告Git。
