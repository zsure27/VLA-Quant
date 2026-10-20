# 035 A1 条件有效性烟雾：环境 seed 与固定初态

状态：预注册，尚未启动。取代同日 v1；v1 没记录 `env.reset()` 后、`set_init_state()` 前的状态。

## 唯一问题

修正真实评测顺序为 `seed_all(model_seed) → env.seed(environment_seed) → env.reset() → set_init_state(official_state)` 后，改变 `env_seed` 是否让冻结评测协议下**策略实际可见**的输入发生可重复的变化？同时记录固定初态注入之前的 reset 观测，判断差异在哪一步消失。

只比较 `env_seed=0/1`。固定 LIBERO Spatial 的 10 个任务、官方初态索引 5、模型 seed 0、等待 10 步、之后固定 no-op 8 步、processor、图像/状态处理及全部物料。每个 task/seed 在新环境重复两次。无策略、无训练、无闭环成功率。

## 事前判定

1. 同 seed 重复的 reset 原始状态、query0 与后 8 步策略输入必须一致或处于预先记录的渲染噪声范围；不得出现初态已成功、提前 done 或环境错误。
2. 只有 query0 的 processor BF16 像素或归一化 proprio 在至少 8/10 任务跨 seed 的差异超过同 seed 噪声，且后 8 步仍可复现，才允许把 `env_seed=1` 当作新的评测条件，进入冻结 C0/C3 各 10 回合配对微型闭环。
3. 若 reset 原始状态变化而策略输入未变，归类为固定 `set_init_state` 覆盖了 seed 效应；若两者均未变，归类为环境 reset 本身未生成有效差异。两种情况都不启动 env-seed C0/C3 长评测，改为另立可观测 reset 条件的小烟雾。
4. BF16 的 9/10 只作历史参考，10/10 不构成门槛；任务 4 原样保留。

这批官方初态及其任务属于历史开发范围。即便本烟雾通过，后续 C0/C3 结果也先报告为条件敏感性证据；训练/开发初态重叠审计未通过前不称独立泛化或盲测。

## 产物

保存修复版评测 helper、probe、评测器、LIBERO 实现、processor 配置与初态数组的前后 SHA256；逐任务保存 reset 原始图像/状态及 qpos/qvel、query0/后 8 步真实策略输入的哈希与差异；保存 runner 退出码、GPU 状态和 gate 结论。
