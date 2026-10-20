# P2.5 三种状态分布对照：peft_train 补测预注册

固定 exact-12L、blocks18–19 的 data80 rank8 Recovery-LoRA（已冻结，不再训练），在 `peft_train` 的**其他帧**上用同观测 BF16 查询并比较 C0 无 LoRA 与 C1 LoRA 的逐步/逐维、夹爪阈值和动作误差。帧选择为 P1 v3 轨迹划分中 peft_train 的每条轨迹 20% 时位，按数据集顺序取首 80 条不同轨迹；与训练时的 80 个校准文件路径不同，显式保存轨迹/帧哈希。原训练校准80只用于重建固定 adapter 结构，完整 adapter 权重由冻结 checkpoint 覆盖。评测不访问 router_dev 或 offline_final_holdout。

结果与已完成的 router_dev 295 帧及本轮学生访问观测比较，但三者的轨迹数、时位采样和自主闭环访问机制不同，均值差异只能提示分布错配，不能单独证明 covariate shift 因果机制。每个分布报告 mean/median、轨迹级汇总、worst-decile、逐step/逐维与夹爪 signed margin，并在可能时按任务分析。若学生访问状态显著恶化且与失败时序一致，再预注册同 rank8、同优化预算的 student-state teacher relabeling；否则优先诊断目标与控制执行错配。P4/P5/Router 保持锁定。
