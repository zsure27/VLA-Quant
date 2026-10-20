# 035 C0/C3 固定环境条件扩展：官方 reset 10–19

前置：修正顺序无策略条件烟雾 `PASS_CONDITION_SMOKE_ONLY`；微型10对 `PASS_POLICY_MICRO`；首片50对 `PASS_FIRST_SHARD_PROTOCOL`，42/50 对45/50，rescue4/break1，区间跨0。首片的收益方向和大小不改变本片既定样本范围。

唯一研究问题：冻结 C3 相对同底座 C0 在同一可观测 `env_seed=1` 条件下的配对收益和 rescue/break 结构，是否在下一个固定开发分片延续？本片为 10 个 Spatial 任务各官方 reset 10–19，共 **100 回合/配置**；不重复 5–9。固定 exact12L AWQ W2A16 mixed backbone、三份 profile、原 C3 rank8 blocks18–19 adapter、model seed0、修正顺序 helper、8步 chunk、评测/成功定义。C0 与 C3 只差是否加载 C3 adapter。服务器 runner 自动 C0 后接 C3，完成即停在分析门禁，不自动排更大片。

启动前/完成后核对冻结源码、checkpoint配置、profile和 adapter SHA；阶段完成核验各100 manifest、无 episode error、逐回合配对、首查询观测/动作 trace、rescue/break、逐任务净差、时间/字节成本。任何物料或协议失效即停止扩展并备份。官方 reset 10–19 仍为历史开发证据；不称独立盲测，不把旧无效 seed1 运行并入当前样本。无需BF16 10/10；不训练新adapter、不启用 Router/P4/P5、不触碰 offline_final_holdout。
