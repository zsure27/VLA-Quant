# 035：C3 在修正后环境随机条件中的配对闭环侧证

日期：2026-09-30
计划：`a1-035-envseed1-10-19-20260930`
阶段终态：`COMPLETE`，revision 3，exit code 0。服务器状态文件最后更新时间：2026-09-30 08:59:38 UTC。

## 结论

本片给出**方向性侧证**：在 10 个 LIBERO Spatial 任务、官方初态索引 10–19 的 100 个配对回合中，冻结的 C3 比 C0 多成功 4 回合；C3 救回 9 个 C0 失败回合，同时使 5 个 C0 成功回合失败。差值为正，但配对置信区间跨 0，精确 McNemar 检验也不显著。因此不能据此宣称 LoRA 的闭环收益已复现或稳定。

这不是新初态数组或盲测。它沿用官方初态数组，在修正过 seed 调用顺序的评测器下改变环境随机条件；更准确的定位是**同一官方初态支持集上的环境扰动开发侧证**。旧 110 `env_seed=1` 结果已判为 `A1_INVALID_CONDITION`，不可与本片合并。本片自己的评测条件通过了预先执行的可见性烟雾门禁，但仍不能替代独立初态、训练/开发重叠审计或锁定 holdout。

## 配对结果

| 数据范围 | C0 | C3 | Rescue | Break | 净差 | 分层 bootstrap 95% 区间 | 精确 McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 本片 reset 10–19，100 对 | 81/100 | 85/100 | 9 | 5 | +4 pp | −2 到 +10 pp | 0.4240 |
| 首片 reset 5–9 与本片合并，150 对 | 123/150 | 130/150 | 13 | 6 | +4.67 pp | 0 到 +10 pp | 0.1671 |

150 对中任务 1 的净改善最大（6 个 rescue）；任务 5、6 分别净增 3、2。退化集中在任务 7（1 rescue、4 break，净 −3），另有任务 3 的 1 个 break；任务 9 有 1 rescue 和 1 break。该异质性说明单一静态 C3 可能并非处处有益，但 rescue/break 的总差只有 7/150，且任务分布很不均。它不足以证明模型能从决策前环境上下文预测哪种适配器更好，更不能作为 task-ID 路由的证据。

本片没有 C2/14L 配对臂，所以 `C2−C0` headroom 与 W4 recovery fraction 均为 `null`。不得用历史 14L、W4 或 BF16 成绩补成同条件恢复率。

## 评测与数据核验

- C0、C3 各完成 100 回合，任务/初态严格配对；reset 索引为 10–19，与首片 5–9 合并后共 150 个唯一 task/reset pair，无重复。
- 两个评测进程 exit code 均为 0。执行语义固定为 8 步 action chunk；C0 记录 1,529 次策略查询，C3 记录 1,471 次。
- C3 是冻结的 exact-12L、rank-8 Recovery-LoRA，只训练 blocks 18–19；本片仅做评测，没有新训练。Adapter SHA256：`67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`。
- 运行物料包括 evaluator、修正后的 LIBERO helper、W2/W4/G64 profile、模型配置与 adapter。评测前后材料清单逐字节相同。关键 evaluator/helper SHA256：`b4efd426d521e0e44438df48b4e8aba39280d9e4c3160e0ac8ca1913d87207b6`、`a4061279eebe82944f048bebfc9e85fa85acf0d7a9cf5d6ce18aea1102694fe2`。
- 原始逐回合结果、3,000 个策略观测 NPZ、trace 与控制台输出共 3,020 个文件、488,612,762 bytes。服务器评测目录与本机归档逐文件 SHA256 tree fingerprint 一致：`98af60e1d7dd434b1ffd3c970cf2d1cbe0864d8cb2b371b8953532b09ee281e0`。日志引用了 200 个 rollout MP4，但这些文件不在评测目录或收尾脚本检查的 rollout 根目录中；其中仅找到 10 个其他运行的视频，未错误归入本轮。视频未包含在已核验副本中，原日志和观测 trace 均保留。
- 未触碰 `offline_final_holdout`。这些 reset 属于历史/开发证据，不能称最终泛化集。

## 对当前 LoRA 与 routing 决策的解释

100 对结果可以作为“blocks 18–19 的 C3 在一种经可见性校验的环境扰动下仍有正向净差”的弱侧证，但统计不确定性仍大。合并 150 对后方向一致，CI 下界刚好落在 0，McNemar `p=0.167`；两批来自相邻官方 reset 索引和同一开发支持集，不等于两份独立复现。

Rescue 9、break 5（合并为 13、6）表示存在一部分双向翻转，尚无证据表明冲突规模“很大”或可由环境路由解决。episode 级结果不能直接做每 8 步的 Router 标签：C0 与 C3 可能很早就进入不同状态，后续同 query 序号并非同一状态。若未来要判断上下文路由，必须在可靠保存/恢复的同状态分支中测量适配器差异，再用决策前 H17/proprio 做跨轨迹预测，并对比 task-ID、instruction-only、打乱上下文和 Always-C3。

## 下一轮候选：扩大单个 Recovery-LoRA 的语言 W2 覆盖

用户提出先测静态微调的能力上限，再决定是否增加专家/路由。这个顺序合理。候选配置 `C3-W2Lang-All-r8` 固定 exact-12L backbone，只把目标从 blocks 18–19 扩到语言侧全部 W2 blocks：`{0–7, 16–19, 24–31}`；保持视觉、12 个语言 W4 blocks、AWQ profile、保护模块不变。目标线性层类型应与 C3 完全一致，不能把语言 W4 层或 BF16 保护模块误纳入。

**单变量边界要如实解释：**每层 rank 仍为 8，训练数据、教师标签、初始化方法、Smooth-L1 配方、1000 步、学习率、优化器、batch 语义和模型种子固定；改变的是 LoRA 覆盖范围，但参数数、adapter 字节、梯度/优化成本会随覆盖同步增加。因此该试验回答“扩展覆盖加相应容量后能恢复多少”，不是纯粹隔离层范围的因果比较。训练前必须枚举实际 target modules，精确报告可训练参数、文件字节、显存及步时；现有 2-block 的 1,249,280 参数不能直接线性外推成最终测量。

建议执行顺序：

1. 只做 CPU/单批次正确性审计：确认 20 个语言 W2 blocks 的 target 列表；冻结主干与保护层；验证梯度只流入目标 LoRA、保存/重载输出一致、参数量和实际 adapter bytes。若目标模块契约不清或误触冻结层，停止。
2. 固定现有 student-state80 样本及 BF16 同观测标签；按相同 peft_train 校准来源为新目标层生成 Response-SVD 初始化。先做低步数训练烟雾，确认 loss finite、目标层更新、非目标权重哈希不变，再做预注册的完整 1000 步训练。
3. 闭环先过 `docs/EVALUATION_SMOKE_GATE_20260929_CN.md` 的真实调用顺序、观测可见性、重复稳定性、配对微型测试。随后同一批配对 reset 上比较 C0、冻结旧 C3 和新 wide-LoRA；首片最多 50 回合/配置后硬 gate。烟雾/协议通过才按预注册计划继续，不能按成功率挑样本或任意延长。
4. 报告总成功率、rescue、break、净差、任务内分布、最差退化、配对区间及真实参数/字节/训练与推理成本。宽覆盖若以较高成本显著改善且 break 很少，优先保留简单单 LoRA、暂不做 Router；若收益有限且翻转确有双向性，也先证明同任务内部的决策前上下文能在未见轨迹上预测差异，再考虑第二个适配器。task ID 或 instruction 单独解释的收益应归类为 task-conditioned selection，不称 contextual routing。

这份扩层试验是下一轮候选，不在本次已完成评测中启动。A1/A2 的有效独立条件、训练/开发重叠与最终泛化门禁仍需单独遵守；纯 W2、P4/P5 和 Router 均未解锁。

## 归档路径

- 服务器原始评测：`/root/autodl-tmp/qvla-repro/eval/a1-035-envseed1-10-19-20260930/`
- 服务器收尾快照（含当前 Git bundle、运行锁、status 与 SHA 清单）：`/root/autodl-tmp/qvla-repro/backups/035-a1-envseed1-20260930-180739-852d350/`
- 快照中的评测树为同盘 hardlink 只读归档路径；源目录与本机完整副本的逐文件 SHA256 tree fingerprint 相同。该快照不替代独立本地副本。
- 本机忽略归档：`backups/experiments/p2-shared-peft/20260930-035-envseed1-10-19/a1-035-envseed1-10-19-20260930/`
- 本机可追踪配对数据与汇总：`results/experiments/p2-shared-peft/20260930-035-envseed1-10-19/`
- 同名分析报告：`reports/experiments/p2-shared-peft/20260930-035-envseed1-10-19/`
