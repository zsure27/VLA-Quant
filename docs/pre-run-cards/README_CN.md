# 实验运行前卡片

本目录保存尚未产生配对结果的预注册方案。卡片不是实验成功率报告，也不单独创建 `reports/experiments/` 与 `results/experiments/` 会话目录。执行后，原始产物、分析和 gate 决定应进入同名的模块化会话目录，并在报告中回链对应卡片。

- [`A1_EVALUATION_REPLICATION_20260929_CN.md`](A1_EVALUATION_REPLICATION_20260929_CN.md)：冻结 C3 的评测条件复现预案，当前为历史 HOLD；`env_seed` 单独变化已被实测判定无效，不能直接按此卡派发 GPU 长实验。

未来声称独立 seed/reset 的长闭环实验先执行 [`EVALUATION_SMOKE_GATE_20260929_CN.md`](../EVALUATION_SMOKE_GATE_20260929_CN.md)。
