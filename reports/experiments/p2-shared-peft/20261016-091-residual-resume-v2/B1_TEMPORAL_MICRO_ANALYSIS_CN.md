# B1 temporal v2 微测分析

两侧micro与strict audit均实际exit0；PASS_B1_TEMPORAL_PAIRED_PROTOCOL，10个配对task/reset5。Early80 9/10，Phase80 3/10；Phase−Early为0 rescue/6 break，净−60pp，任务经验bootstrap95%[−90,−30]pp，exact McNemar p=.03125。仅小片同十任务开发，重叠正式，不加入分母、不挑checkpoint、不据方向改变续片，不外推未见任务。

新增失败（Phase相对Early）：task0/1/4/7/8/9的reset5；Early唯一失败task5/reset5，Phase失败另含task5/reset5。不从标签推断物理机制。全部逐对标签见同名JSON。

真实两侧首query双相机/指令/归一化一次本体（原capture float32存储表示）与原PTQ对应task0/reset5完全一致；零B 8×7动作按原capture float32逐位一致；恢复训练后动作保存重载相同、冻结SHA36d41eb329862e8c8a6f302a88655b729b85ebe750aaf93dc32de92b2ba984b6与训练一致。严格audit核验实际argv/运行lock/主机/seed/首观测/全部原始NPZ动作/finite/无异常/20原视频完整覆盖。修复获得真实GPU证据，旧失败v1仍排除。

成本：Early147查询/274.734371秒，Phase234查询/311.397009秒，包含加载、SHA、真实query门禁、trace、video，不解释为INT2部署加速。各侧19988480参数，训练及独立教师成本另见训练报告。

当前唯一原v2 plan已自动first50-early80（05:38实测revision12）；GPU3MiB是当次加载早期观测，策略54607 CPU136%，不据此判停。随后Phase50→strict audit。任何失败停止；首50完成后再全量分析/双份/新鲜额度与盘/独立后片注册。活动EV源码、runtime-lock、已锁报告不覆盖，微测归档只显式完成根，排除活动首50。
