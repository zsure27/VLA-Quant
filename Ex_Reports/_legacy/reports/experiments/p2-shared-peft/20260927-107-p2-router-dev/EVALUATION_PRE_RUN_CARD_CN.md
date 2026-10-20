# A/B/C/D 轨迹级开发门禁：预运行卡

日期：2026-09-27。类型：MODEL-SELECTION / DEVELOPMENT。主阶段 P2.5；P4/P5/Router 锁定。

问题与假设：固定观测 data80 收益中的覆盖与训练步数作用尚混淆。完整 A=16/200、B=16/1000、C=80/200、D=80/1000 四格，在**同一** router_dev 59轨迹/295帧上配对比较后，能分离两主效应及交互，并识别最差轨迹退化。此次只是固定评测，不再训练。

唯一变量：已训练 rank8 LoRA 状态对应的轨迹覆盖与步数两因子；teacher、exact12L baseline、观测、评测代码、AWQ profile、seed 和动作语义固定。`router_dev` manifest SHA256 `0d42a53a07412f2839e0c30aee0ae606e914a3e3707748f1a577ba8e2370d364`；v3 split SHA256 `3560425653ef4b297a48a3854aae20348898129b6673db1474943e95897606a2`。offline_final_holdout 封存。

主指标：按轨迹中位数的 teacher 动作误差差值、改善轨迹比例、最差十分位退化。次指标：frame mean/median、逐维 position/rotation/gripper、夹爪阈值分歧与 margin，模型状态字节和运行时间。历史32帧只作回归调试，不参与开发门禁。

正结果的决定：若 B、C 或 D 的轨迹级分布比 A 有稳健改善且无关键夹爪/最差轨迹退化，仅选一个最有信息量的候选做后续受控闭环；若仅训练步数改善，优先检查目标/优化量；若仅覆盖改善，优先检查数据与学生访问状态。负结果则冻结 compute/coverage 扩展，回到同观测 H1/H2 机制诊断。

停止规则：严格 1 个 teacher、1 个 baseline、4 个已固定候选；不得新开 rank/loss/SVD 变体；总 GPU 时间上限 2 小时。每阶段有退出码/状态，失败只修复同一计划且不得重做完成阶段。原始输出与分析保存同名 results/reports，本机完整归档和服务器持久盘保留；GitHub 固定 zsure27/VLA-Quant。
