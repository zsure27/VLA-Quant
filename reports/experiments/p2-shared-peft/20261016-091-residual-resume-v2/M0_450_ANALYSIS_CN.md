# M0 reset40–49及正式450分析（2026-10-17）

五正式片全部完成；最后三阶段exit0/revision11/PASS_B3_INIT_ONLY_FULL_SHARD_PAIRED_PROTOCOL。恢复首50的47+3分段证据保持，不伪造旧退出码。实际命令/锁/seed/首观测/每NPZ SHA/动作/单次本体归一化/finite、首query零分支/PTQ/冻结/保存重载及各片原始视频门禁通过。

## 结果与不确定性

本片PTQ18/100、S040/100；25救回/3新增失败，净22.0000pp，任务经验聚类95%[6.0000,41.0000]pp，McNemar p=2.7440488e-05。

正式reset5–49各450：PTQ77（17.11%）、S0184（40.89%）；119救回/12新增失败，净23.7778pp，任务经验聚类95%[9.5556,39.1111]pp，McNemar p=2.5870039e-23。8任务净改善、0任务净变差。

当前冻结A4条件下，仅训练前响应SVD补偿无需优化已有闭环恢复，但存在新增失败，仍明显没有恢复到既有训练后S的历史表现。后一句只是不同历史运行的描述，不是同轮严格配对差值；不得把旧046 S426/450与本轮标签拼接或据此计算因果训练增益。S0初始与PTQ的差异证明该配方下残差补偿有效，尚不证明动作敏感新方向创新；普通LoRA普遍失效也未获证明。

训练80诊断中Z1000动作MSE下降88.18%，不能把其历史闭环低成功解释为完全未学会。训练误差与闭环分布/长时误差累积/初始化条件之间的机制需要单变量检验，标签不能识别具体物理失败。单suite十任务多次开发，非盲测/未见任务；micro与正式重叠不加，holdout角色保持。

## 失败、互补与成本

本片新增失败：task3/reset44, task3/reset47, task3/reset48。

本片ptq：2557查询/1762.698秒/100视频/0优化步。四个100片累计成本：{"queries": 10191, "wall_seconds": 7073.909424543381, "scope": "Four completed100-episode shards; recoveredfirst50 mixed old/new attempt timing remains separately in report"}

ptq本片全部失败：task0/reset40, task0/reset41, task0/reset42, task0/reset43, task0/reset44, task0/reset45, task0/reset46, task0/reset47, task0/reset48, task0/reset49, task1/reset40, task1/reset41, task1/reset42, task1/reset43, task1/reset44, task1/reset45, task1/reset46, task1/reset47, task1/reset48, task1/reset49, task2/reset40, task2/reset45, task2/reset48, task3/reset41, task3/reset43, task4/reset40, task4/reset41, task4/reset42, task4/reset43, task4/reset45, task4/reset46, task4/reset47, task4/reset48, task4/reset49, task5/reset40, task5/reset41, task5/reset42, task5/reset43, task5/reset44, task5/reset45, task5/reset46, task5/reset47, task5/reset48, task5/reset49, task6/reset40, task6/reset41, task6/reset42, task6/reset43, task6/reset44, task6/reset45, task6/reset46, task6/reset47, task6/reset48, task6/reset49, task7/reset40, task7/reset41, task7/reset42, task7/reset43, task7/reset44, task7/reset45, task7/reset46, task7/reset47, task7/reset48, task7/reset49, task8/reset40, task8/reset42, task8/reset43, task8/reset44, task8/reset46, task8/reset47, task8/reset48, task8/reset49, task9/reset40, task9/reset41, task9/reset42, task9/reset43, task9/reset44, task9/reset45, task9/reset46, task9/reset47, task9/reset48, task9/reset49

本片svd-init-only：2177查询/1600.648秒/100视频/0优化步。四个100片累计成本：{"queries": 8629, "wall_seconds": 6430.078627824783, "scope": "Four completed100-episode shards; recoveredfirst50 mixed old/new attempt timing remains separately in report"}

svd-init-only本片全部失败：task0/reset44, task1/reset40, task1/reset41, task1/reset42, task1/reset43, task1/reset44, task1/reset45, task1/reset46, task1/reset47, task1/reset48, task1/reset49, task3/reset41, task3/reset44, task3/reset47, task3/reset48, task4/reset40, task4/reset42, task4/reset43, task4/reset46, task4/reset48, task4/reset49, task5/reset40, task5/reset42, task5/reset43, task5/reset44, task5/reset45, task5/reset46, task5/reset47, task5/reset48, task5/reset49, task6/reset40, task6/reset41, task6/reset42, task6/reset43, task6/reset44, task6/reset45, task6/reset46, task6/reset47, task6/reset48, task6/reset49, task7/reset41, task7/reset42, task7/reset44, task7/reset46, task7/reset48, task7/reset49, task8/reset43, task8/reset44, task8/reset46, task8/reset47, task8/reset48, task9/reset41, task9/reset42, task9/reset43, task9/reset44, task9/reset45, task9/reset46, task9/reset47, task9/reset48, task9/reset49

完整450逐键成败、全部救回/新增失败/任务差值/累计失败见同名results/cumulative-450-analysis.json；首50恢复成本含原873.9968秒中断尝试与新181.0165秒三个回合，其旧13查询保留但不加正式查询分母。全部墙钟含初始化/校验/trace/video IO，不是纯推理延迟或真实INT2部署加速。S0额外80独立校准轨迹/历史SVD构造成本单列；适配器19988480参数/39976960 BF16 tensor字节，0优化不等于零信息成本。

## 下一阶段与数据保存

先完成最后片200视频双份归档及完整阶段索引，再审查B1 Early80/Phase80同40A4完整轨迹的内容覆盖；源不足必须按原训练初态完整重采并保持数据角色，不能替用B5双策略数据。新方法先独立预注册/CPU/实际输入数值门禁再GPU，不改已完成M0/source336。动作敏感方向与动作目标分别单变量，SQ有自己的等价/梯度门禁。当前额度窗口1792183963首次15/10延期仍有效，不能按时间推断04:52刷新；最终收尾遵循双份/Git/091原生/PAUSED/总结/Windows正常关机顺序。
