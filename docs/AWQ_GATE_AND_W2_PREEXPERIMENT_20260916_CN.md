# AWQ W4 闭环门槛与 W2 语言剪裁预实验（2026-09-16）

本轮使用 AutoDL 014、OpenVLA-OFT LIBERO Spatial、相同的量化入口和检查点。在十个任务中固定各五个官方初始状态、相同 model/env seed 和每状态 SHA-256；每种策略跑 50 个闭环回合。状态 0 复现先前十状态筛选，状态 1–4 是新增的 40 个状态。运行时的 evaluator 未改动；`base-commit.txt` 与 `evaluator.patch` 保存了实际代码状态，`command.txt`、`SHA256SUMS.txt` 和完整 EVAL/console 日志留在 `results/experiments/p0-foundation-baselines/20260916-awq-baseline-gate/`，大模型与 profile 保留在 AutoDL 持久盘。

| 配对集合 | 同入口 BF16 | 模拟 AWQ W4A16 | W4－BF16 |
| --- | ---: | ---: | ---: |
| 全部 50 状态 | 47/50 (94%) | 47/50 (94%) | 0 pp |
| 新增 40 状态 | 38/40 (95%) | 39/40 (97.5%) | +2.5 pp |
| 先前重复的状态 0 | 9/10 | 8/10 | −10 pp |

全 50 个状态中 BF16 独有成功 1 次，W4 独有成功 1 次；对这组任务固定、只重抽已观察初始状态的分层 bootstrap 差值 95% 区间为 [−6,+6] 个百分点，精确 McNemar 双侧 p=1。这些数字支持当前模拟 W4 闭环线路继续作为 W2 对照，但并不证明两者等价，也不能证明真实 packed AWQ、模型体积、预热同步延迟或峰值显存都合格。任务 4 的两条线路都是 3/5，任务 9 都是 4/5；同一状态是否成败不同应查看 `paired_episodes.csv`，不能只看任务总数。

先前十状态检查中，全 W2 为 0/10，仅语言 W2（去语言剪裁、视觉保持 BF16）为 1/10，仅视觉 W2（语言保持 BF16且不应用语言 block scales）为 10/10。该证据把短预实验放到语言端，不能据此宣称视觉量化无损。旧 AWQ 块审计对保留的三个诊断样本与五个选取层固定输入，显示 W2 去剪裁的局部动作 token hidden relative MSE 常高于带剪裁 W2，而 W4 低得多；对 224 个语言 target 的整体因果影响不能从三样本的局部 MSE 推断，热图只负责筛选对照。审计原始 `metrics.json`、manifest、teacher 恢复标记和生成脚本保存在 `results/experiments/p0-foundation-baselines/20260916-awq-block-audit/`。

十状态“仅语言 W2 + 原始 W2 profile 剪裁”闭环对照已完成：**0/10**，与旧去剪裁语言-only **1/10** 相比并无改善。视觉保持 BF16、同一 422-target profile 合约、W2 G128 block scales 和固定种子；两种语言侧候选只改变 clip 的应用，先前视觉 G64 候选在语言-only 范围没有施加视觉 quant。十个状态不足以精确比较 0/10 与 1/10，但已能排除“剪裁局部 MSE 较小即闭环恢复”这一简单解释。逐状态 manifest、初始状态哈希和种子已核对相等，两条线路使用相同 W2 基础 profile，均应用 224 个语言目标；去剪裁候选移除了 160 项语言 clip。原始日志、命令、退出码、评估器 patch、哈希清单和逐任务对照位于 `results/experiments/p0-foundation-baselines/20260916-awq-w2-language-clip10/`。去剪裁及视觉 G64 是 PTQ 搜索，不是 PEFT。进一步训练可以只优化冻结整数码的敏感组 log dequant scale（Scale-PEFT）或在冻结骨干上增加低秩 BF16 recovery LoRA；不同任务的 scale 专家是否互补必须先用交叉任务矩阵验证，之后才有理由测试 Contextual Routing。

旧端到端 32 个诊断样本的离线动作 MSE 则从语言 W2 带剪裁的 4.0257 降到语言 W2 全去剪裁的 0.05633，8 步夹爪动作分歧计数从 149/256 降到 10/256；旧 `summary.json` 已复制到 `results/experiments/p0-foundation-baselines/20260916-awq-block-audit/e2e-interventions-old-summary.json`，完整旧产物仍位于服务器持久目录 `artifacts/awq-e2e-interventions-20260912-173315-1136/`。这与新闭环 0/10 对 1/10 并不矛盾：离线样本并非这十条完整的交互轨迹，较小动作误差不足以保证状态反馈、抓取和释放成功。十状态差别只出现在任务 8 的状态 0，下一次选择 PEFT 候选不能只按隐藏层 MSE 排序，应先展示同观测动作、夹爪决策和失效阶段，再以独立初始状态做闭环门槛。

仍未做的门槛包括 W4 相同观测的 GT/teacher action L1、动作 token hidden cosine、预热后的同步延迟和峰值显存；当前评价器的初始 trace 位置存在观测状态可能已被策略调用更改的问题，此轮图不使用该错误标签。W3 profile、真实打包 quant parity、OFT 原基座与 LoRA 合并身份均没有验证，不可将它们写成已完成结果。
