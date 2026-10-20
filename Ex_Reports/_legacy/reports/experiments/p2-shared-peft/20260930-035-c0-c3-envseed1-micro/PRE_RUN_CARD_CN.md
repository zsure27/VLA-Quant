# 035 C0/C3 配对微型闭环预注册

状态：待运行。本卡依据同轮 `a1-035-envseed-order-fix-smoke-v2-20260930` 的无策略烟雾：10/10 任务的 query0 与后 8 步策略图像输入跨 `env_seed=0/1` 改变，同 seed 重复完全一致，初态有效。无策略门禁只允许微型闭环，未证明 C3 的收益。

唯一研究问题：冻结 C3 Recovery-LoRA 在这个可观测的新环境随机条件下，相对相同 12L backbone 的 C0 是否仍有配对闭环收益；记录 rescue（C0 失败/C3 成功）与 break（C0 成功/C3 失败）。C0/C3 各 10 回合，10 个 Spatial 任务各官方 reset 索引 5 一次，model seed 0、env seed 1；所有任务包括历史 BF16 失败的 task 4。除了是否加载既有 C3 adapter，两配置的模型、12L profile、评测器、seed、动作语义、任务和初态均相同。只报告开发性条件敏感性，训练/开发 reset 重叠未厘清前不称盲测或独立最终泛化。

运行前核验 evaluator/helper/恢复代码、C3 adapter、三份 profile、checkpoint 配置和初态 SHA，实际 helper 必须为修正顺序版本；记录实际命令。服务器顺序执行 C0 后自动接续 C3，不重复已完成片。完成后核对 10+10 回合、无异常、逐回合 task/reset/model/env seed/初态哈希严格配对、首 query 观测和动作 trace、与旧 env_seed=0 轨迹是否逐字节相同。成功标签相同本身不能判为条件无效。微型回合结果无论正负，都不得筛任务或据此改首片样本量。

若协议通过，下一计划最多先扩至每配置 50 回合并停在硬 gate；后续长片须另行预注册并核验物料、manifest、trace 与额度。若协议失败，停止扩展并保留全部产物。BF16 10/10 不是门槛；本阶段不训练 adapter、不改 backbone、不运行 Router/P4/P5、不触碰 offline_final_holdout。
