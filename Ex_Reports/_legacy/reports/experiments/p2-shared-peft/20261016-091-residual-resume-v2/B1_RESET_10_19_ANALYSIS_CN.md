# B1 reset10–19 与累计150完整分析

## 本片严格门禁

三阶段全部真实exit0/revision11，PASS_B1_TEMPORAL_FULL_SHARD_PAIRED_PROTOCOL，完整100配对。实际argv/运行lock/主机/seed/全部初观测/NPZ动作/finite/无异常/原始视频覆盖通过，两侧真实首query同task0/reset10 PTQ双相机/任务/一次proprio及零B动作匹配、训练后重载与冻结SHA通过。

Early98/100，Phase46/100；Phase−Early0救回52新增失败，净−52pp，任务经验95%[−73,−31]pp，exact McNemar p4.440892098500626e-16，8任务净变差/0净改善。

## 累计正式150

Early146/150(97.333333%)，Phase65/150(43.333333%)。Phase−Early0救回81新增失败，净-54.000000pp；任务经验95%[-74.666667,-32.666667]pp，exact McNemar p8.27180612553028e-25，0任务净改善/8净变差。

仅同十任务新初态开发；微测重叠不加，失败v1无admitted query不加，不与M0或046跨分母合并。条件/训练1000固定，不挑checkpoint，不因方向调整续片，不由标签推断物理机制。唯一训练变量为取帧时间；Early训练教师逐位与父cache一致，Phase独立教师。

## 成本

本片Early1359query/1239.461317秒，Phase2107query/1583.711535秒；累计Early2052query/1950.038989秒，Phase3218query/2475.740654秒，包含加载/校验/数值gate/trace/video，不称INT2部署加速。每侧19988480参数/39976960 BF16 tensor字节；训练/独立教师及旧失败v1额外成本另列训练和失败报告。

## 全部累计逐对成败

|task|reset|Early|Phase|Phase相对事件|
|---|---|---|---|---|
|0|5|True|False|新增失败|
|0|6|True|True|同成|
|0|7|True|False|新增失败|
|0|8|True|False|新增失败|
|0|9|True|False|新增失败|
|1|5|True|False|新增失败|
|1|6|True|False|新增失败|
|1|7|True|False|新增失败|
|1|8|True|False|新增失败|
|1|9|True|False|新增失败|
|2|5|True|True|同成|
|2|6|True|True|同成|
|2|7|True|True|同成|
|2|8|True|True|同成|
|2|9|True|True|同成|
|3|5|True|True|同成|
|3|6|True|True|同成|
|3|7|True|True|同成|
|3|8|True|True|同成|
|3|9|True|True|同成|
|4|5|True|False|新增失败|
|4|6|True|True|同成|
|4|7|False|False|同败|
|4|8|True|True|同成|
|4|9|True|False|新增失败|
|5|5|False|False|同败|
|5|6|True|False|新增失败|
|5|7|True|False|新增失败|
|5|8|True|False|新增失败|
|5|9|True|False|新增失败|
|6|5|True|True|同成|
|6|6|True|True|同成|
|6|7|True|False|新增失败|
|6|8|True|True|同成|
|6|9|True|False|新增失败|
|7|5|True|False|新增失败|
|7|6|True|True|同成|
|7|7|True|False|新增失败|
|7|8|True|False|新增失败|
|7|9|True|False|新增失败|
|8|5|True|False|新增失败|
|8|6|True|True|同成|
|8|7|True|False|新增失败|
|8|8|True|False|新增失败|
|8|9|True|True|同成|
|9|5|True|False|新增失败|
|9|6|True|False|新增失败|
|9|7|True|False|新增失败|
|9|8|True|False|新增失败|
|9|9|True|False|新增失败|
|0|10|True|False|新增失败|
|0|11|True|False|新增失败|
|0|12|True|True|同成|
|0|13|True|False|新增失败|
|0|14|True|False|新增失败|
|0|15|True|True|同成|
|0|16|True|True|同成|
|0|17|True|True|同成|
|0|18|True|True|同成|
|0|19|True|True|同成|
|1|10|True|False|新增失败|
|1|11|True|False|新增失败|
|1|12|True|False|新增失败|
|1|13|True|False|新增失败|
|1|14|True|False|新增失败|
|1|15|True|False|新增失败|
|1|16|True|False|新增失败|
|1|17|True|False|新增失败|
|1|18|True|False|新增失败|
|1|19|True|False|新增失败|
|2|10|True|True|同成|
|2|11|True|True|同成|
|2|12|True|True|同成|
|2|13|True|True|同成|
|2|14|True|True|同成|
|2|15|True|True|同成|
|2|16|True|True|同成|
|2|17|True|True|同成|
|2|18|True|True|同成|
|2|19|True|True|同成|
|3|10|True|True|同成|
|3|11|True|True|同成|
|3|12|True|True|同成|
|3|13|True|True|同成|
|3|14|True|True|同成|
|3|15|True|True|同成|
|3|16|True|True|同成|
|3|17|True|True|同成|
|3|18|True|True|同成|
|3|19|True|True|同成|
|4|10|True|False|新增失败|
|4|11|True|False|新增失败|
|4|12|True|True|同成|
|4|13|True|True|同成|
|4|14|True|True|同成|
|4|15|True|True|同成|
|4|16|True|True|同成|
|4|17|True|True|同成|
|4|18|True|True|同成|
|4|19|True|False|新增失败|
|5|10|True|False|新增失败|
|5|11|True|True|同成|
|5|12|True|False|新增失败|
|5|13|False|False|同败|
|5|14|True|False|新增失败|
|5|15|True|False|新增失败|
|5|16|True|False|新增失败|
|5|17|True|False|新增失败|
|5|18|True|False|新增失败|
|5|19|True|False|新增失败|
|6|10|True|False|新增失败|
|6|11|True|False|新增失败|
|6|12|True|True|同成|
|6|13|True|True|同成|
|6|14|True|False|新增失败|
|6|15|True|True|同成|
|6|16|False|False|同败|
|6|17|True|False|新增失败|
|6|18|True|True|同成|
|6|19|True|True|同成|
|7|10|True|False|新增失败|
|7|11|True|False|新增失败|
|7|12|True|True|同成|
|7|13|True|False|新增失败|
|7|14|True|True|同成|
|7|15|True|False|新增失败|
|7|16|True|False|新增失败|
|7|17|True|False|新增失败|
|7|18|True|True|同成|
|7|19|True|True|同成|
|8|10|True|False|新增失败|
|8|11|True|False|新增失败|
|8|12|True|False|新增失败|
|8|13|True|False|新增失败|
|8|14|True|True|同成|
|8|15|True|True|同成|
|8|16|True|False|新增失败|
|8|17|True|False|新增失败|
|8|18|True|False|新增失败|
|8|19|True|True|同成|
|9|10|True|False|新增失败|
|9|11|True|False|新增失败|
|9|12|True|False|新增失败|
|9|13|True|False|新增失败|
|9|14|True|False|新增失败|
|9|15|True|False|新增失败|
|9|16|True|False|新增失败|
|9|17|True|False|新增失败|
|9|18|True|False|新增失败|
|9|19|True|False|新增失败|

Early全部失败：task4/reset7, task5/reset5, task5/reset13, task6/reset16。
Phase全部85个失败、81新增失败及完整配对另见B1-cumulative-150-analysis.json，无失败遗漏。

## 续片

本片完整200原视频/原始query/参数/源码双份SHA、Git远端核验后，只按协议与新鲜额度/盘>=8GiB/GPU闲注册reset20–29。前gate/本报告SHA锁定，续片只变允许字段及同reset PTQ首query参考路径SHA，目标正式各450，不按成绩方向选条件。此分析未来锁定不得追加归档段，归档回执另独立报告。
