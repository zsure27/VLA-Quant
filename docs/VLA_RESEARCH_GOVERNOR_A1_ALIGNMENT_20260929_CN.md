# Research Governor v2 对齐：先复现共享恢复，再讨论路由

更新：2026-09-29。本文将本轮上传的 Research Governor 落到当前仓库证据上；执行前以本轮 [A1 PRE-RUN CARD](../reports/experiments/p2-shared-peft/20260929-a1-evaluation-replication/PRE_RUN_CARD_CN.md) 的门禁为准。当前只规划 A1，**未放行 GPU 实验**。

## 1. 目前可以说什么

主底座仍是 `awq_w2a16_12l_mixed_spatial_v1.json` 的 12L 混精 AWQ-W2A16，C3 是只作用于语言 blocks18–19 的 student-state80 rank8 Recovery-LoRA。上轮在开发 reset5–49 的 450 个配对回合中，C0=368、C1=366、C2=388、C3=389；C3−C0=+21，救回35、新增失败14，配对区间约 +1.56 至 +7.78 个百分点。C3 在此协议上与 14L 相近，**尚未证明跨评测条件或训练种子稳定，也未证明优于 14L**。task1/5 的净改善合计23，超过总净改善21；task6/7 退化。该异质性是机制问题，不是 Router 已有收益。

“student-state80”是 80 个学生访问的策略查询观测，每个由同观测 BF16 产生 8×7 教师动作；它不是 80 条训练轨迹。旧 demo-only C1 与 C3 的差别主要是训练观测分布。C3 的正收益支持分布对齐的重要性，但由于仅一个优化 seed、同一开发初态集，不能证明 covariate shift 是唯一原因。旧 12L 411/412、14L 430/431 属跨 session 漂移；继续保留原始口径并单独做来源审计，不拿它们拼出新的恢复率。

## 2. A1 的关键收窄：先证明“新条件”确实新

A1 只回答：**冻结 C3 后，相对 C0 的闭环收益能否在独立评测条件复现？** 五个比较对象 BF16、C0、C1、C2、C3 均用同一新条件重新运行，不能拿旧 seed 的控制组与新 seed 的 C3 配对。模型、权重、量化 profile、评测器、8 步执行与模型 seed 均固定；候选仅改变 `env_seed`。分析单位是 episode，保留任务层异质性与所有配对翻转，不能用动作 MSE 替代成功率。

源码审计揭示一个可能让 A1 失去意义的陷阱：评测器按索引加载官方 50 个固定初态，`env.seed` 不改变 `initial_state` 数组；随后 `set_init_state` 将环境放到该固定状态。因此 `env_seed=1` 是否产生**独立的可观测评测条件**尚无证据。运行前先做不使用 GPU 的双种子同索引检查：比较初始观测，并用同一固定动作序列比较后续环境状态，保存哈希和差异。若只有数值噪声或没有差异，就停止此种子方案；不能仅凭 seed 数字不同声称独立复现，也不能为了救这个方案同时改变 reset 和扰动。新的 reset 或扰动需另立单变量协议、来源与训练/开发重叠审计，并保持 `offline_final_holdout` 封存。

第二个前置条件是物料完整性：本机归档已重新核对 C3、活动代码和原始结果包；上轮契约记载三份 profile、旧 C1 adapter 和模型 checkpoint，但并非所有原始二进制都在本机。新实例必须逐项复算实际加载文件，与冻结 SHA 比对。路径相同、文件名相同或 Git HEAD 相同不能代替二进制哈希。若 checkpoint 没有旧 SHA，先从可信原始归档/此前 manifest 建立可追溯参照；无法做到就不运行 A1。

A1 的支持标准在运行前冻结：C3−C0 净差为正、配对 CI 下界大于零，至少两个任务净改善，同时独立报告 task1/5、task6/7 及 BF16 差距。净差消失或反转则否定此 seed 下的稳定性；正差但 CI 跨零属不确定。无论 A1 结果如何，都不直接进入 Router。A1 支持后，A2 仅改变 LoRA 训练 seed，80 个训练观测、教师标签、rank8、blocks18–19、初始化政策、1000步、损失、优化器和评测协议保持不变。

## 3. 路由主张需要比“事后能救回”更强的证据

A1/A2 稳定后，先检查 C0 与 C3 是否在**同一任务内部**随当前观测发生可预测的优劣切换。静态参照是 `Always C0`、`Always C3`；随后计算 Task-ID 固定选择上界，再评估真正的状态或 chunk 选择空间。独立 rollout 的同 query 序号并不代表同一观测；二者已经分叉时，不能据此造“逐状态 oracle”。同观测 BF16 差值只称 *local teacher preference*，不能称闭环最优标签。真正的 chunk 行为上界需要可靠的 simulator 状态保存/恢复与同状态分支，若做不到则将该上界标为未知。

只有观察到超出 task 固定选择的稳定同任务状态切换，且 block17 后、block18 前的因果 H17/当前 proprio 对这种切换有独立评测预测力，才讨论第一版 binary adapter gate：每次 8 步 query 决定 C3 LoRA OFF/ON。它必须优于 Always C3 和 Task-ID/instruction-only 简单规则；若收益仅来自任务标识，只能称 task-conditioned gating，不能宣称环境驱动的 Contextual Routing。正负翻转本身可能来自分布失衡、关键动作或夹爪阈值，并非自动说明两个专家互补。C2 多专家与动态 W2/W4 profile 必须经过各自的互补性和额外字节/切换成本门禁。

## 4. 后续研究顺序与终止点

1. **A1**：固定当前 C3 的独立评测条件复现。当前 HOLD；先完成新条件与哈希门禁。若无法建立单变量、非重叠、可观察的新条件，停止 GPU 工作并报告方法学缺口。
2. **A2**：仅在 A1 可解释后，用独立训练 seed 复现 C3；不得同步更换样本、教师、loss 或评测 seed。共享收益不稳定则回到学生状态采样和训练分布，不开始路由。
3. **B**：从 paired same-observation 与可行的同状态分支中量化 `Best Fixed → Task Oracle → State/Chunk Oracle` 的额外空间；离线教师距离与真实闭环行为分开标注。
4. **C1**：B 有稳定同任务状态空间后，先试 C0/C3 的 OFF/ON 最小门控，比较 Always C3、Task-ID、instruction-only、proprio、H17、H17+proprio、打乱上下文与随机门控，计入路由延迟和字节。
5. **C2/D**：只有 C1 或 B 支持可复现的修复冲突才考虑两个等预算 LoRA 专家；只有动态精度有独立于静态 W4 的上下文上界且存储成本合理，才考虑切换 profile。
6. **E/F**：方法成立后按预注册精度退火逐步减少 W4 保护，逐级重采集 H17、重测基线和成本；SmoothQuant 保持独立扩展，先过 smoothing-only 数值等价。

每一步均先填 PRE-RUN CARD。负结果应结束或重定向一个明确假设，而不是自动触发更多 rank、SVD、clip、Scale 或静态 W4 组合搜索。最终评价并列报告闭环成功、配对不确定性、真实权重/adapter/profile 字节、GPU 成本与每个 chunk 的推理延迟；AWQ fake quant 不得被描述为已证明 packed INT2 加速。
