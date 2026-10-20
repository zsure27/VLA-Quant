# P2.5 同学生访问状态的无 LoRA 控制：预注册

已完成的 exact-12L/LoRA/14L 配对 Spatial500 为 411/408/431。LoRA 的 7,642 次学生访问查询由冻结 BF16 同观测重标后，raw action MSE 均值 0.03260、夹爪分歧 9.17%；离线 peft_train 其他帧80 和 router_dev295 的 LoRA raw MSE 均值约 0.02125/0.02206。三分布取样和回合长度不同，不能仅凭均值宣布 covariate shift 已被证明。

本实验固定此前 LoRA 策略产生的 500 条开发回合，每回合只取第 4 次因果策略查询（`query_in_episode=3`，执行约 24 步后）；五片每片 100 个观测，已预检全部存在。对相同观测运行冻结 BF16 和 exact-12L **无 LoRA**，与既有 LoRA 同观测动作及 BF16 回答严格配对。唯一模型变量为是否加载已冻结的 blocks18–19 rank8 LoRA；不使用回合未来成败作为策略输入，不训练、不触碰 holdout。预先比较 raw 8×7 MSE、逐维/逐步、夹爪 0.5 阈值符号与 margin、LoRA 改进回合比例，并按闭环成功/失败和任务分层。新 BF16 答案应与已保存的同状态 BF16 数值等价；若不等价，停止解释并审计 evaluator/样本契约。

这仍是诊断：无 LoRA 策略没有从这些状态独立 rollout，不能由同状态探针推导反事实闭环成功。若 LoRA 在其访问状态降低 BF16 误差而闭环仍未改善，优先诊断目标重要性与 8 步执行；若它在学生访问状态失去离线收益，才进一步考虑同 rank8、同预算的 student-state teacher relabeling。P4/P5/Router 继续锁定。
