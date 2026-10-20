# P2 Shared PEFT 报告（107，2026-09-25）

本轮只在 `awq-w2a16-12l-mixed-spatial-v1` 上修改语言 blocks 18–19。正式配方为
DINO W2/G64、SigLIP W2/G128、语言 W4 blocks 8–15 与 20–23，其余语言 W2/G64；
14L 的 431/500 仅作为把 blocks 18–19 静态升为 W4 的恢复参照。

训练数据来自 P1 的 `peft_train` 完整轨迹切分。第一轮 pilot 使用 16 条互异训练轨迹，
统一训练 200 步；离线开发比较使用历史/开发帧，只用于判断是否值得进入有限闭环，不能
代替闭环成功率。

## 配置审计

首次离线 pilot 的通用诊断组合只对 DINO 使用了 G64，语言 W2 仍为 G128。该问题由
blocks 18–19 的 Scale 参数量 3,162,112 与 P1 G64 契约的 6,324,224 不一致而发现。
这些运行保留为配置不匹配的诊断证据，不进入 P2 gate，也不能冠以 12L 主底座结果。

修正后直接调用与静态闭环相同的 `attention_visual_stage_plan`。真实 profile 审计得到
422 个量化目标、32 个语言 block scale、84 个 W4 Linear；blocks 18–19 全部为 W2/G64。

## 结果

| exact-12L 配置 | 归一化动作 MSE | 原始动作 MSE | 夹爪分歧 | 可训练参数 |
|---|---:|---:|---:|---:|
| 12L baseline | 0.09668 | 0.03458 | 0.08203 | 0 |
| rank8 LoRA，标准零输出 | 0.07516 | 0.02511 | 0.08594 | 1,249,280 |
| rank8 LoRA，Response-SVD | 0.07461 | 0.02474 | 0.08594 | 1,249,280 |
| fixed-code Scale-PEFT | **0.06340** | **0.02275** | **0.07813** | 6,324,224 |

Scale-PEFT 同时改善两种动作 MSE 和夹爪分歧，通过离线开发 gate；两个 LoRA 虽改善连续
动作误差，但夹爪分歧略升，暂不进入闭环 gate。当前正在重跑 Scale-PEFT 以保存 adapter
state，随后只做有限闭环开发验证。shared PEFT 未在留出闭环上稳定改善前，不进入专家
互补性或 Contextual Routing。
