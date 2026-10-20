# 035 环境 seed 修正烟雾结果

结论：`PASS_CONDITION_SMOKE_ONLY`。这只证明修正后的 `env_seed=0/1` 在官方初态索引 5 上产生了可重复、可进入策略输入的环境差异；尚未证明 C3 的闭环收益，也不是新初态盲测。旧 110 A1 的 `A1_INVALID_CONDITION` 结论继续保留。

## 证据

| 指标 | 结果 |
| --- | ---: |
| 完成任务 | 10/10 |
| 每任务每 seed 新环境重复 | 2 次；共 40 次初始化 |
| 同 seed 的 reset 原始状态、query0 与后 8 步输入精确重复 | 10/10 任务 |
| seed 变化使 `env.reset()` 后、固定初态前观测改变 | 10/10 任务 |
| seed 变化使 query0 的真实 processor 输入改变 | 10/10 任务 |
| 后 8 步仍有真实 processor 输入差异 | 10/10 任务 |
| query0 BF16 processor 图像不同值比例 | 最小 16.65%，平均 18.83%，最大 21.06% |
| 初态已成功、等待/固定动作提前 done、环境错误 | 均为 0 |
| 前后物料 SHA 清单 | 一致 |
| GPU | 前后均 1 MiB、0% 利用率；未加载策略 |

同一固定官方初态下，query0 的双视角图像在 10/10 任务改变，归一化 proprio 在 0/10 任务改变；sim qpos 在 7/10 任务改变。LIBERO 的固定初态注入会重设仿真状态，但 reset 中的部分场景变化仍留在可见图像内。源码显示 reset 采样也可改变模型中的固定物体位置，因此这是一个合理机制解释；确切的物体级归因尚未完成。

旧评测 helper 先调用 `env.seed`，随后 `seed_all(model_seed)` 重新设置 NumPy，导致环境 seed 被覆盖。修正版 helper SHA256 为 `a4061279eebe82944f048bebfc9e85fa85acf0d7a9cf5d6ce18aea1102694fe2`；探针 SHA256 为 `9111b6c49fa1108a8d523d701f3cf64dfa1c7e48676e095e9243dd2651fa11ee`。实际导入路径与 `model_seed → environment_seed` 顺序已断言，运行退出码为 0。原始逐任务哈希/差值在同名 [`results`](../../../../results/experiments/p2-shared-peft/20260930-035-envseed-order-fix-v2/)；完整服务器产物保留在 `/root/autodl-tmp/qvla-repro/eval/a1-035-envseed-order-fix-smoke-v2-20260930/`，本机副本位于忽略目录 `backups/experiments/p2-shared-peft/20260930-035-envseed-order-fix-v2/`。

## 研究判断与下一步

本次仅放行冻结 C0/C3 各 10 回合的配对微型闭环；必须核验真实 evaluator 命令、模型/profile/adapter 哈希、10 对 manifest、首 query 观测和动作 trace，计算 rescue 与 break。当前测试继续使用已研究过的官方 reset 5，训练/开发 reset 重叠尚未完全审计，故后续结果最多是新的**可观测环境条件**上的开发性证据。BF16 历史 9/10 只是参考，task 4 保留，10/10 成功率不是条件有效性门槛。长评测需另过首片最多 50 回合/配置的协议硬门禁。
