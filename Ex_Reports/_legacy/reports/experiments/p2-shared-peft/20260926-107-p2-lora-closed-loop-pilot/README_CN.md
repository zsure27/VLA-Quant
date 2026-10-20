# P2 Recovery LoRA 配对闭环开发评估（107，2026-09-26）

## 实验问题

在冻结的 exact-12L AWQ-W2A16 backbone 上，只给仍为 W2 的语言 blocks 18–19 加载 rank8 Recovery LoRA，检验 200 步端到端动作蒸馏带来的离线动作误差改善能否转化为 Spatial 配对闭环成功率。

本轮使用 Response-SVD 初始化 checkpoint。C0 是 12L，无 adapter；C1 是相同 12L 加 Recovery LoRA；C2 是将 blocks18–19 升为 W4 的 14L 静态参照。三个配置使用相同 seed、环境 seed 和 states0–4，每个配置 50 回合。该范围按协议只属于历史/开发证据。

## 结果

| 配置 | 成功数 | 成功率 |
|---|---:|---:|
| C0：12L | 43/50 | 86% |
| C1：12L + rank8 LoRA | 42/50 | 84% |
| C2：14L static | 43/50 | 86% |

C0 与 C1 的配对四格为：共同成功 41、仅 C0 成功 2、仅 C1 成功 1、共同失败 6；McNemar 精确检验 `p=1.0`。C1-C0 的配对 bootstrap 95% 区间为 `[-8,+4]` 个百分点。当前样本中 C2 没有高于 C0，因此静态恢复分母非正，恢复比例没有可解释值。

## 分析

Response-SVD LoRA 在冻结离线集上把 normalized action MSE 从约 0.09668 降到 0.07323，但有限闭环中比相同 12L 少成功 1 回合。结果没有显示稳定正收益，置信区间也覆盖实质退化与小幅改善。它复现了 Scale-PEFT 已观察到的模式：降低 BF16 教师动作 MSE 不能单独保证闭环恢复。

本轮的主要不确定性是 states0–4 规模小、属于开发证据，且 C0/C2 在该分片恰好同为 43/50。由于效应方向不是正向，直接扩大相同 checkpoint 的回合数缺少收益证据；继续消耗闭环预算也不能修正训练目标和部署目标之间的错配。

LoRA checkpoint 为 28 个张量、1,249,280 个参数、2,509,154 bytes。只在 blocks18–19 挂载，backbone、量化配置、评测器和配对协议未改变。原始日志、策略查询、命令和哈希保存在服务器与本机完整归档；Git 只保存小型配对结果和分析。

## Gate 结论与下一步

Recovery LoRA 未通过 shared PEFT 闭环 gate。结合 full Spatial500 的 Scale-PEFT 结果（407/500，低于同协议 12L 的 411/500），当前两种 shared PEFT 都没有稳定留出闭环收益，因此不进入 P4 专家互补或 P5 Contextual Routing。

下一项若继续 P2，应先版本化训练配方并只改变一个主要变量，优先检验 action chunk/闭环敏感的训练目标，而不是扩大当前 checkpoint。新配方必须与相同参数预算的简单 LoRA 对照，并先做有限开发闭环；通过稳定正收益后才能扩展独立初态或进入专家门禁。

## 证据路径

- Git 小结果：`results/experiments/p2-shared-peft/20260926-107-p2-lora-closed-loop-pilot/`
- 本机完整归档：`backups/experiments/p2-shared-peft/20260926-107-p2-lora-closed-loop-pilot/`
- 服务器原始目录：`/root/autodl-tmp/qvla-repro/eval/p2c-lora-closed-loop-pilot-0-4-20260926-040207-82285`
- Adapter SHA256：`8408908bc73becc547c955d4c37be220c468599af1a15c228d435436cea4b953`

