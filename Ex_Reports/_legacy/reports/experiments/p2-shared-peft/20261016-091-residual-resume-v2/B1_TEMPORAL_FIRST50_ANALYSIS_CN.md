# B1时间覆盖首50完整分析

Early80 48/50(96%)，Phase80 19/50(38%)。Phase−Early：0救回、29新增失败，净−29/50=−58pp；任务经验bootstrap95%[−78,−36]pp，exact McNemar p=3.725290298461914e-9，0任务净改善/8任务净变差/2任务相同。仅同十任务开发，不是未见任务泛化或独立环境复现。微测重叠不加分母，旧失败v1无有效query排除。

两侧首50与严格audit实际exit0，六阶段终态revision20；PASS_B1_TEMPORAL_PAIRED_PROTOCOL。实际argv/运行lock/主机/派生seed/每episode初始观测及NPZ/动作finite/连续8步执行/无异常/100原视频严格通过。真实首query双相机/任务/一次proprio按capture float32原表示与原PTQ相同，零B动作逐位相同，保存重载/全冻结SHA与训练一致。无放宽数值容差。

唯一实验变量仍Early80与Phase80取帧时间，其余40训练轨迹/80样本/初始化/19988480参数/1000优化步/教师协议相同；不能从两个不同训练样本的loss比较推出收益，不挑checkpoint、不按方向改变续片；不从成败标签推断物理失败机制。

## 成本

Early693查询/710.577672秒，Phase1111查询/892.029120秒；含加载/SHA/真实query门禁/trace/video，不称INT2部署加速。每侧19988480参数/39976960 BF16 tensor字节，完整训练和独立教师成本见训练分析。旧失败v1额外运行成本与10异常视频单独保留，不加入正式成功率。

## 全部配对及失败

|任务|reset|Early|Phase|相对Phase事件|
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

Early失败task4/reset7和task5/reset5；Phase所有31失败均列上表，29新增失败另见完整JSON break_pairs。

## 后续

完成本片完整双份后，按协议/新鲜额度/盘>=8GiB、运行留6GiB顺序reset10–19/20–29/30–39/40–49各100/侧，总各450。后片独立版本化审计与注册器，复用已冻结EV2真实loader；只变offset/count/output/lock和对应同reset PTQ参考路径SHA；固定训练/seed/算法/数据不变。每片前gate/完整分析SHA锁定，不扩routing、不读holdout、不因当前方向提前筛选或停候选。
