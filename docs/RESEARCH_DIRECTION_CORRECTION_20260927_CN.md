# 研究方向自纠偏决定：从离线恢复转向闭环因果诊断

日期：2026-09-27。依据：[Research Governor 原文](VLA_RESEARCH_GOVERNOR_20260926.txt)、[P2.5 方案](P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md)、[data80 配对报告](../reports/experiments/p2-shared-peft/20260926-107-p2-data80-stability-and-realignment/README_CN.md)。本文件是阶段决定；若后续原始证据与此冲突，先审计来源并新增版本，不覆写旧结论。

## 当前主问题和证据边界

**PRIMARY=P2.5，LOCKED=P4/P5/P6。** 研究目标是判断冻结 exact-12L AWQ-W2A16 底座上的紧凑 shared Recovery-LoRA 能否修复闭环动作能力；若不能，先区分训练目标/状态分布错配 H1 与真实上下文修复冲突 H2。只有 H2 的可复现证据或 shared PEFT 的可信闭环收益，才能申请 P4；只有冻结专家之间存在双向互补并可由推理前 H17 预测，才能申请 P5。

2026-09-26 data80 试跑：LoRA 42/50、12L 43/50、14L 43/50。LoRA 对 12L 的配对净差为 -1/50，McNemar p=1，bootstrap 95% CI 为 [-0.10,+0.06]；该 slice 的静态 W4 headroom 为 0，W4 恢复率为 null。固定观测 MSE 0.02284、31/32 帧改善是机制证据，没有证明闭环恢复。历史 32 帧已反复使用，只保留回归调试用途。data16→data80 同时改变轨迹覆盖 16→80 和训练步数 200→1000，标记 **CONFOUNDED RESULT**；B=16/1000 与 C=80/200 是分离两因子的最小补充控制。A/B/C/D 先走 trajectory-level router_dev，不批量送闭环。

12L 的 411/412 与 14L 的 430/431 差异尚无完整来源审计。当前只说“约 411/500”和“约 430/500”；源 session、评测器提交、checkpoint/profile、初态 manifest、seed、G64/G128 均核对后再指定 CANONICAL/HISTORICAL/SUPERSEDED。本次 P2.5 同观测诊断固定具体文件哈希，因此不依赖上述争议的精确 500 回合口径。

## 现在允许回答的问题

1. **同观测闭环对齐（已预注册并启动）：**保存学生自身访问的观测和动作，再让冻结 BF16 查询同一观测。逐 episode、逐 query、逐 8 步位置和 7 个动作维度比较，报告夹爪阈值与 margin、H17、首次分歧和成功。历史开发 reset 的 10 回合只作 DIAGNOSTIC，不作为最终成功率；大分歧阈值在 router_dev 冻结前为 null。正结果支持 H1 的进一步受控试验；负结果转向动作语义、时间位置和目标错配检查，不自动增加模型容量。
2. **轨迹级 router_dev：**只用 peft_train 作 Response-SVD 校准，按轨迹固定抽样并保存帧哈希；先建立 exact-12L 与已训练 LoRA 的 frame/trajectory 分布，报告中位数、改善轨迹比例、最差退化和夹爪行为。offline_final_holdout 保持封存。
3. **计算与覆盖分离：**在同 rank8、Response-SVD、Smooth-L1、优化器、LR、batch 语义和 blocks18–19 下补 B/C；先用 router_dev 判定是否需要有选择的闭环验证。离线指标此时是变量归因控制，不能代替闭环门禁。
4. **若 H1 被支持：**预注册同参数预算的 student-state BF16 同观测重标，对比 offline-only 和 student-state-distilled；只在配对闭环与成本证据支持时主张 shared PEFT 恢复。
5. **若 H1 已处理仍失败：**以梯度余弦、残差方向、Response-SVD 子空间和跨上下文 adapter response 检查 H2。方向对齐则停止 Router 动机；稳定冲突也只解锁 P4 互补性试验的提案，不直接训练 Router。

## 每次实验的硬门禁

每个 GPU 实验先保存 `PRE_RUN_CARD`：阶段、因果假设、**一个**改变变量、固定控制、数据切分、主指标、正反两种结果分别改变的决定、停止条件、预计成本、具体输出路径和哈希。将实验明确归为 CONTRACT、DIAGNOSTIC、DEVELOPMENT 或 FINAL。完成后保存 `POST_RUN_DECISION`：支持/不支持/不确定、允许与不允许的因果主张、阶段与分支状态、唯一下一问题。若正反结果都只导向另一个相似变体，该实验不得启动。探索要设最大变体、GPU 时间和停止规则。

当前冻结 exact-12L：DINO W2 G64、SigLIP W2 G128、语言 W4 blocks8–15与20–23，blocks18–19保持 W2；PEFT 只修改这两个 block，rank8。Scale-PEFT 是负基线。静态 W4 搜索、Scale 调参、rank/SVD 网格、更多固定 32 帧目标调整、专家与 Router 不符合当前准入门禁。新 backbone 需要版本化配置与配对理由，不能静默改动。

实例已开启时仍遵守额度与连续运行规则：不因等待研究方向而让 GPU 空跑；有合规且预注册的 P2.5 项就接续，若只剩跨门禁选择或无科学上合理项则备份并关机。15%启动归档，10%停止新项并快速收尾，至少留3%；完整备份须有原始产物、哈希、配对分析、局限、成本和下一步决定。
