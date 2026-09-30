# 035 C0/C3 配对微型闭环结果

门禁：`PASS_POLICY_MICRO`，仅表示可按已预注册的 reset 6–9 补到首片 50 对。C0 与 C3 在修正 seed 顺序、可观测的 `env_seed=1` 条件下各完成 10 回合；两者均为 **9/10**，`rescue=0`、`break=0`、净差 0。两者共同失败的是 task 1。这 10 对中没有观察到 C3 的闭环恢复收益，样本量及较高基线成功率不足以据此否定或确认先前 450 回合的开发性 C3 候选。

## 配对与执行核验

- 10 个 Spatial 任务均使用相同的官方 reset 索引 5、相同的派生 model/env seed 和初态哈希；C0/C3 各自退出码 0，0 episode error。实际命令除 C3 adapter 与输出目录外完全一致，12L W4 block 列表为 8–15、20–23。
- 修正后 helper、评测入口、LoRA 加载代码、C3 adapter 和三份 profile 的运行前后 SHA 清单一致；adapter SHA256 为 `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`，helper 为 `a4061279eebe82944f048bebfc9e85fa85acf0d7a9cf5d6ce18aea1102694fe2`。
- 首次策略查询的观测哈希在 10/10 对中相同；C3 首个原始动作 chunk 在 10/10 对中与 C0 不同，表明 adapter 已参与推理。所有查询均 finite，8 步 chunk 语义正常。逐回合结果、事件/动作 trace 哈希及门禁审计保存在同名 [`results`](../../../../results/experiments/p2-shared-peft/20260930-035-c0-c3-envseed1-micro/)；完整观测 `.npz` 原件保留在服务器 `/root/autodl-tmp/qvla-repro/eval/a1-035-c0-c3-envseed1-micro-20260930/` 和本机忽略目录 `backups/experiments/p2-shared-peft/20260930-035-c0-c3-envseed1-micro/`。

## 局限与下一步

本 slice 只包含每任务一个历史开发初态。当前新条件确实进入模型输入，但训练/开发 reset 重叠尚未完整审计；不能称独立盲测。BF16 先前的 9/10 来自另一批新生成初态，不能与本 slice 的 C0/C3 9/10 混成同一配对比较。14L 在该条件未测，W4 恢复率为 `null`。

按固定计划运行官方 reset 6–9 的 C0/C3 各 40 回合，与本片合成首片 50 对；之后停止在协议硬 gate，分析逐任务 rescue/break、异常、观测和动作 trace、成本。不得根据这 10 对结果选择任务、调整预定样本量或直接启动 Router。
