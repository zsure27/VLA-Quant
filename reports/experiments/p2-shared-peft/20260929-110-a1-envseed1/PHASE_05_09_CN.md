# 110 A1 环境种子复现：首片阶段分析

**状态：阶段结果，仅 10 任务 × 官方 reset5–9，共 50 个配对回合/配置；不能判定 A1。**模型、适配器、profile、评测器、模型 seed 与初态索引均冻结，环境 seed 从历史开发条件 0 改为 1。固定动作双次预检查表明同 seed 首次观测哈希可重复，而跨 seed 策略可见图像变化；它验证的是同初态不同环境随机条件，**没有新初态数组，也不是盲测**。

| 配置 | 成功 |
|---|---:|
| BF16 | 50/50 |
| C0 exact-12L | 42/50 |
| C1 旧 demo-only rank8 LoRA | 41/50 |
| C2 14L 静态 W4 参照 | 44/50 |
| C3 student-state80 rank8 LoRA | 43/50 |

C3−C0=+1/50：3 个救回，2 个新增失败；按任务分层的配对 bootstrap 95% 区间为 −6 至 +10 个百分点，精确 McNemar p=1。任务 1 增 1、任务 5 增 2，任务 6 和 7 各退 1，其余净差为 0。C2−C0=+2/50；仅两个回合的 headroom 使任何恢复比例都极不稳定，不用于推进门禁。BF16 50/50 提示该切片偏易，仍需其余预注册 reset5–49。

五组经 `task_id`、`init_state_index`、`model_seed`、`env_seed` 和 `init_state_sha256` 严格配对，均无 episode error，原始小结果在同名 `results/experiments/p2-shared-peft/20260929-110-a1-envseed1/paired-05-09-*.json*`。服务器原件在 `/root/autodl-tmp/qvla-repro/eval/a1-110-envseed1-5-9/`。本机副本 SHA256：摘要 `c1d17fe0534e84441f878a90cff3fd5434d296ee0b1866efc36ea8c3abb5f5e5`，逐回合 `d5c5d4dc13bf0559bd977726b56726a79147fc22ae2bd0487bf7e4c9730abb14`。

运行故障：初始 BF16 命令携带混精候选而被参数检查拒绝；v2 又保留了多余 G64/W4 profile 参数，亦在启动前被拒绝。两次失败均为零 BF16 回合；v3 使用独立 BF16 旁路命令补跑 50 回合，原失败日志保留在 BF16 结果目录，其他四组未重复运行。此修正只改变 BF16 的无效调用形式，没有更换评测条件。

下一步为已登记的 reset10–19 同条件五配置配对评测；只在五小时额度允许时继续。后续还需检查环境 seed 导致的图像变化具体来自何种环境随机性，避免把纯渲染变化误称新物理初态。Router、P4/P5、新 adapter 训练和 `offline_final_holdout` 保持锁定。
