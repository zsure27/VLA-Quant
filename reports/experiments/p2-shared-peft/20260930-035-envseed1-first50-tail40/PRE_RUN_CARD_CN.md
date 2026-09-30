# 035 A1 首片 50 对：余下 40 对预注册

前置门禁：无策略条件烟雾 `PASS_CONDITION_SMOKE_ONLY`；C0/C3 各 10 回合（每任务官方 reset 5）的配对微型闭环协议通过，首 query 观测 10/10 相同、首动作 10/10 不同、0 episode error；其 9/10 对 9/10、rescue=0、break=0 只作结果记录，不用于改变本片的样本选择。

本片只运行同一 10 个 Spatial 任务的官方 reset 6–9，每配置 40 回合；与已完成的 reset 5 各 10 回合合并成 reset 5–9 的首片 50 对，不重复 reset 5。冻结修正后 evaluator/helper、exact12L 三份 AWQ profile、C3 rank8 Recovery-LoRA、model seed 0、env seed 1、8 步 chunk、任务顺序和成功定义。C0/C3 唯一区别仍是是否加载已有 C3 adapter。服务器顺序 runner 在 C0 完成后自动接 C3，终态只进入首片硬 gate，不自动派发更长片。

首片硬 gate 核验真实命令与源码/模型/profile/adapter哈希、C0/C3 各 40 回合的全部 manifest、与微型10对合并后50对无重复、首 query 观测和动作 trace、环境条件的可见性与执行成本。成功率正负不决定是否放行或调整预定后续样本；若条件/协议有缺陷即停止并归档。所有结果归类历史官方 reset 上的新可观测环境条件开发证据，训练示范重叠未审计完成前不称盲测或最终泛化。本片不训练新 adapter、不改 backbone、不启用 Router/P4/P5、不触碰 offline_final_holdout。
