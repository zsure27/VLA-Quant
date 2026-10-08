# PRE-RUN CARD：B3 all-eligible W2 recovery V2

PRIMARY STAGE：P2 shared PEFT recovery；LOCKED：P4/P5/Router。

QUESTION：同一A4底座上，新语言32-block rank8 LoRA能否恢复其闭环能力？

HYPOTHESIS：低秩语言残差可修复部分W2表征损失；若视觉/不可低秩误差主导，静态语言修复仍不足。

BACKBONE HASH：版本化A4配置及代码SHA见执行包；DINO W2/G64、SigLIP W2/G128，语言全部32块W2/G64，V/O无额外clip、MLP保留clip；保护模块高精度。无W4岛。

METHOD：新224个语言Linear的rank8/448矩阵；Response-SVD只用peft_train校准80帧，视觉完全冻结。不把B1 20-block状态迁移充数。AdamW1e-4/weight_decay0.01、Smooth-L1 beta0.1/8×7、batch1/1000步、seed/order_seed7。

DATA SPLIT：A4新采集10任务×reset0–3×query0/1，40回合80观测，不按成功筛选；冻结BF16同观测标签。开发reset20–24排除训练reset；offline_final_holdout不查询。

ONE CHANGED VARIABLE：在固定A4上新增语言adapter。B3−A4才是主因果差值；B3−B1同时变底座和scope，只有描述意义。

CONTROL：A4；同条件参照B1、BF16、A0。

PRIMARY METRIC：B3−A4的闭环成功净差。

SECONDARY METRICS：rescue、break、任务聚类95%区间、McNemar、与参照差距、失败模式及参数/字节/秒/显存。

EXPECTED DECISION：A4上正恢复且成本合理→冻结候选，判断离目标还有多大差距；恢复不足→冻结此单一配置、分析瓶颈；不能为了Router目标强造多个adapter。

STOP CONDITION：A4状态缺失/非有限/来源错误、数据重叠、W4目标出现、覆盖缺失、代码或SHA变化、冻结/零输出/梯度/重载失败、协议失败、单阶段10800秒或实时额度门限。MAX_VARIANTS=1，MAX_GPU_TIME由整计划6小时墙钟上限约束。问题需审阅时报告并关付费实例。

OUTPUT DIRECTORY：新实例同名归档下`B3/`；同名reports/results分析。

ESTIMATED COST：A4训练状态40回合、Response-SVD校准80帧（spool按目标维度×80帧×56个token×BF16两字节计算，另留6GiB）、烟雾10步、正式1000步、五配置各10+50开发回合。真实显存/秒数须实机烟雾核验；首片之后无自动长片。

ADMISSION：CPU契约就绪，真实GPU烟雾待执行；不称整模型每个参数均2bit，不称独立复现。详见[执行卡](../B2_B3_NEXT_BOOT_RUNBOOK_20261008_CN.md)。本卡替代V1当前执行说明，保留旧记录。
