# BF16 烟雾门槛更正记录

日期：2026-09-30。此记录在 10 回合 BF16 烟雾完成后新增，保留原始 PRE-RUN CARD，不回写为事前预注册。

## 更正原因

原卡把“BF16 在每个候选 reset 上都成功”写成继续检查条件的门槛。这把**reset/评测流程是否有效**与**BF16 策略在该任务上是否成功**混成一个判断。BF16 也可能因任务难度或策略本身而失败；要求 10/10 会把策略表现误用作环境条件有效性的代理量，也会诱发删换失败 reset。

原 BF16 烟雾已完成 10/10 个任务，9 成功、1 失败，退出码为 0、没有 episode error。失败是任务 4 的真实闭环失败（28 次 action query，未标记 aborted），不是 reset 无效或程序故障。保留这次 9/10 原始结果；不替换任务、不重跑失败项，也不把 10/10 作为 C0/C3 的准入门槛。

## 正确的 gate

继续条件只检查：评测器/helper、checkpoint、profile、adapter 和 reset 内容哈希匹配；reset 未预先完成；相同环境随机条件可重复；环境 seed 改变后，真实评测顺序下的场景及模型可见输入出现超过同 seed 重复噪声的变化；任务状态合法且评测命令/trace 语义正确。BF16 成功标签只作参考数据。

下一步主问题是冻结 C3 相对 C0 的配对闭环收益及 `rescue`/`break` 数量。C0 与 C3 必须在相同 task、reset、环境 seed 和模型 seed 下配对；所有预登记任务都保留，包括 BF16 失败的 task 4。先做环境 seed 顺序的无策略探针，再做 C0/C3 微型配对 smoke。微型 smoke 只检查协议和物料，不据其正负挑任务或调整长测样本量。

## 已发现的独立代码问题

真实评测 helper 的原顺序是 `env.seed(environment_seed) -> seed_all(model_seed) -> env.reset()`。本机及 035 使用的 LIBERO `ControlEnv.seed` 调用 `numpy.random.seed`；后续 `seed_all` 又重置 NumPy，因此在 reset 前环境随机流被覆盖。已在独立 overlay 中改为 `seed_all(model_seed) -> env.seed(environment_seed) -> env.reset()`。只有后续同 seed 重复稳定、不同 env seed 的策略输入确实不同，才算这个修复生效；代码改动本身不等于条件通过。

此前准备的每任务 fresh reset 数组、BF16 9/10 结果，以及环境 seed 1 的旧 A1 失败运行各自保留，不能互相冒充为修复后的 seed 复现。旧 A1 仍是 `A1_INVALID_CONDITION`；reset 来源与训练示范的重叠审计仍未完全解决，所以此轮若继续只称修复后条件探针/开发性 C0-C3 配对复现，不称盲测或独立最终 holdout。
