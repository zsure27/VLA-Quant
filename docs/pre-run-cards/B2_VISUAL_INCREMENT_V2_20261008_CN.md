# PRE-RUN CARD：B2 visual increment V2

PRIMARY STAGE：P2 shared PEFT recovery；LOCKED：P4/P5/Router。

QUESTION：固定A3+冻结B1后，新增视觉rank8微调能否带来值得成本的闭环增益？

HYPOTHESIS：剩余部分误差来自视觉W2表征，固定语言修复后视觉残差可减少失败；也可能已无可稳定恢复headroom。

BACKBONE HASH：A3配置SHA见执行包`SOURCE_SHA256.json`，三profile与B1 SHA固定在model_registry_v1.json；实际全checkpoint由profile内容identity复核。

METHOD：B1全部冻结；双塔196个已量化transformer Linear的独立rank8零输出adapter（不含两个patch卷积）；AdamW1e-4/weight_decay0.01，Smooth-L1 beta0.1、8×7，batch1/1000步、seed/order_seed7。

DATA SPLIT：原B1 student-state80（40学生回合、80观测；reset0–3），BF16同观测标签；开发reset20–24；SVD继承来源审计；不触碰holdout。

ONE CHANGED VARIABLE：新增可训练视觉adapter；A3/B1/profile/data/训练语义冻结。预算增加，不能称同预算胜出。

CONTROL：B1；同条件参照A3、BF16、A0。

PRIMARY METRIC：B2−B1的闭环成功净差。

SECONDARY METRICS：rescue、break、任务聚类95%区间、McNemar、任务失败与参照差距、参数/字节/训练秒/峰值显存；不能仅凭净差判断可路由性。

EXPECTED DECISION：有开发收益且成本合理→冻结候选并准备独立确认；无收益/高break→保留B1，冻结该视觉分支；协议失败→修复或终止，不按成功率改门槛。

STOP CONDITION：物料/来源/冻结/shape/finite/重载失败，微测/首片协议失败，单阶段10800秒，或实时额度门限。MAX_VARIANTS=1，MAX_GPU_TIME由整计划6小时墙钟上限约束。不等待用户审阅而付费空跑。

OUTPUT DIRECTORY：新实例`backups/experiments/p2-shared-peft/YYYYMMDD-INSTANCE-b2-b3/B2/`，报告/小结果在同名reports/results。

ESTIMATED COST：10步结构烟雾、1000步正式训练、五配置各10+50开发回合；首片50为统计样本，微测不叠加。真实秒数/显存未知，烟雾实测后评估额度；不能预先承诺4090可容纳。无自动长片。

ADMISSION：CPU契约就绪；真实GPU烟雾待执行。代码/物料与运行顺序见[执行卡](../B2_B3_NEXT_BOOT_RUNBOOK_20261008_CN.md)。本卡替代V1当前执行说明，保留旧门禁历史。
