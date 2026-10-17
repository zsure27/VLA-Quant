# B1 reset20–29与累计250完整分析

## 严格协议与范围

三阶段全部真实exit0/revision11，2026-10-16T23:59:25.738140UTC完成，PASS_B1_TEMPORAL_FULL_SHARD_PAIRED_PROTOCOL/100/offset20/count10。实际argv/运行lock/seed/全部初观测及NPZ动作SHA/finite/异常/原始视频覆盖严格审计通过；两侧真实首query对应task0/reset20 PTQ双相机/指令/一次proprio原capture float32表示/零B动作逐位匹配，训练后保存重载和全冻结均过。

本片Early95/100、Phase45/100；Phase−Early 0救回/50新增失败，净-50.000000pp；任务经验95%[-70.000000,-31.000000]pp，exact McNemar p=1.77635683940025e-15。

## 累计正式250

Early241/250(96.400000%)，Phase110/250(44.000000%)；Phase−Early 0救回/131新增失败，净-52.400000pp，任务经验95%[-72.400000,-32.000000]pp，exact McNemar p=7.3468396926393e-40；0任务净改善/9净变差。

仅同十任务新初态开发；micro重叠不加，失败v1无admitted query不加，不与M0/046跨分母合并。训练1000步及checkpoint固定，不按方向改变续片条件，不据成败标签推断物理机制。唯一训练变量是取帧时间，Early教师与父cache逐位一致，Phase独立教师。

## 全部成本

early80：本片1419query/1262.448749s；累计3471query/3212.487738s。
phase80：本片2100query/1576.147069s；累计5318query/4051.887723s。

含加载/SHA/数值gate/trace/video，不称INT2部署加速；19988480参数/39976960 BF16 tensor字节每侧，训练/教师/失败v1额外成本另列既有报告。

## 本片Early全部失败

[{"task_id": 5, "init_state_index": 21, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 22, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 26, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 27, "early80": false, "phase80": false}, {"task_id": 9, "init_state_index": 21, "early80": false, "phase80": false}]

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
|0|20|True|True|同成|
|0|21|True|True|同成|
|0|22|True|False|新增失败|
|0|23|True|True|同成|
|0|24|True|False|新增失败|
|0|25|True|True|同成|
|0|26|True|True|同成|
|0|27|True|True|同成|
|0|28|True|True|同成|
|0|29|True|True|同成|
|1|20|True|False|新增失败|
|1|21|True|False|新增失败|
|1|22|True|False|新增失败|
|1|23|True|False|新增失败|
|1|24|True|False|新增失败|
|1|25|True|False|新增失败|
|1|26|True|False|新增失败|
|1|27|True|False|新增失败|
|1|28|True|False|新增失败|
|1|29|True|False|新增失败|
|2|20|True|True|同成|
|2|21|True|True|同成|
|2|22|True|True|同成|
|2|23|True|True|同成|
|2|24|True|True|同成|
|2|25|True|True|同成|
|2|26|True|True|同成|
|2|27|True|True|同成|
|2|28|True|True|同成|
|2|29|True|True|同成|
|3|20|True|True|同成|
|3|21|True|True|同成|
|3|22|True|True|同成|
|3|23|True|True|同成|
|3|24|True|True|同成|
|3|25|True|True|同成|
|3|26|True|True|同成|
|3|27|True|False|新增失败|
|3|28|True|True|同成|
|3|29|True|True|同成|
|4|20|True|False|新增失败|
|4|21|True|True|同成|
|4|22|True|True|同成|
|4|23|True|True|同成|
|4|24|True|True|同成|
|4|25|True|False|新增失败|
|4|26|True|True|同成|
|4|27|True|False|新增失败|
|4|28|True|True|同成|
|4|29|True|False|新增失败|
|5|20|True|False|新增失败|
|5|21|False|False|同败|
|5|22|False|False|同败|
|5|23|True|False|新增失败|
|5|24|True|False|新增失败|
|5|25|True|False|新增失败|
|5|26|False|False|同败|
|5|27|False|False|同败|
|5|28|True|False|新增失败|
|5|29|True|False|新增失败|
|6|20|True|False|新增失败|
|6|21|True|True|同成|
|6|22|True|False|新增失败|
|6|23|True|False|新增失败|
|6|24|True|False|新增失败|
|6|25|True|True|同成|
|6|26|True|True|同成|
|6|27|True|True|同成|
|6|28|True|False|新增失败|
|6|29|True|True|同成|
|7|20|True|False|新增失败|
|7|21|True|False|新增失败|
|7|22|True|False|新增失败|
|7|23|True|False|新增失败|
|7|24|True|True|同成|
|7|25|True|False|新增失败|
|7|26|True|False|新增失败|
|7|27|True|False|新增失败|
|7|28|True|False|新增失败|
|7|29|True|True|同成|
|8|20|True|False|新增失败|
|8|21|True|False|新增失败|
|8|22|True|False|新增失败|
|8|23|True|True|同成|
|8|24|True|True|同成|
|8|25|True|True|同成|
|8|26|True|False|新增失败|
|8|27|True|False|新增失败|
|8|28|True|True|同成|
|8|29|True|True|同成|
|9|20|True|False|新增失败|
|9|21|False|False|同败|
|9|22|True|False|新增失败|
|9|23|True|False|新增失败|
|9|24|True|False|新增失败|
|9|25|True|False|新增失败|
|9|26|True|False|新增失败|
|9|27|True|False|新增失败|
|9|28|True|False|新增失败|
|9|29|True|False|新增失败|

## 每任务累计净差

{"0": -10, "1": -25, "2": 0, "3": -1, "4": -9, "5": -18, "6": -11, "7": -18, "8": -15, "9": -24}

## 全部累计失败

### early80 共9个

[{"task_id": 4, "init_state_index": 7, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 5, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 13, "early80": false, "phase80": false}, {"task_id": 6, "init_state_index": 16, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 21, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 22, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 26, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 27, "early80": false, "phase80": false}, {"task_id": 9, "init_state_index": 21, "early80": false, "phase80": false}]

### phase80 共140个

[{"task_id": 0, "init_state_index": 5, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 8, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 5, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 6, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 8, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 5, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 7, "early80": false, "phase80": false}, {"task_id": 4, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 5, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 6, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 8, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 5, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 8, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 5, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 8, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 5, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 6, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 7, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 8, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 9, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 13, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 14, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 12, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 13, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 14, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 15, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 16, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 17, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 18, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 19, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 19, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 12, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 13, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 14, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 15, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 16, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 17, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 18, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 19, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 14, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 16, "early80": false, "phase80": false}, {"task_id": 6, "init_state_index": 17, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 13, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 15, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 16, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 17, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 12, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 13, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 16, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 17, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 18, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 10, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 11, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 12, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 13, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 14, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 15, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 16, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 17, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 18, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 19, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 22, "early80": true, "phase80": false}, {"task_id": 0, "init_state_index": 24, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 21, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 22, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 23, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 24, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 25, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 26, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 27, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 28, "early80": true, "phase80": false}, {"task_id": 1, "init_state_index": 29, "early80": true, "phase80": false}, {"task_id": 3, "init_state_index": 27, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 25, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 27, "early80": true, "phase80": false}, {"task_id": 4, "init_state_index": 29, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 21, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 22, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 23, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 24, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 25, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 26, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 27, "early80": false, "phase80": false}, {"task_id": 5, "init_state_index": 28, "early80": true, "phase80": false}, {"task_id": 5, "init_state_index": 29, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 22, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 23, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 24, "early80": true, "phase80": false}, {"task_id": 6, "init_state_index": 28, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 21, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 22, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 23, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 25, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 26, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 27, "early80": true, "phase80": false}, {"task_id": 7, "init_state_index": 28, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 21, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 22, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 26, "early80": true, "phase80": false}, {"task_id": 8, "init_state_index": 27, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 20, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 21, "early80": false, "phase80": false}, {"task_id": 9, "init_state_index": 22, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 23, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 24, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 25, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 26, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 27, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 28, "early80": true, "phase80": false}, {"task_id": 9, "init_state_index": 29, "early80": true, "phase80": false}]

