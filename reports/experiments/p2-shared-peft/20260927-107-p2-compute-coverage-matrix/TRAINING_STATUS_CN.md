# B/C 训练阶段状态

A 与 D 原始训练仍分别在 2026-09-26 的服务器持久盘归档。B=16轨迹/1000步已完整训练并生成 rank8 LoRA 状态，SHA256 `cfbcd8a29fe39b430902f9b4aec1ee309b1365bbc154b2429d8c73708335d5db`；C=80轨迹/200步完成并生成状态，SHA256 `ea1d66345930c15a5e9efc55e3cd94eb74c840ff50266fdfe0f4e280c0615b9d`。两者都为 exact12L、blocks18–19、Response-SVD、Smooth-L1 beta0.1、LR1e-4、seed7。B 的最终后处理路径错误已单独记录，训练产物保留，C 使用修复脚本完成。

**STAGE STATUS：stay P2.5；BRANCH STATUS：continue only the frozen 2×2 comparison。** A/B/C/D 的因果解释尚未完成：必须在同一 trajectory-level router_dev 上配对比较，并明确 frame/trajectory 聚合、失败轨迹、夹爪和成本。旧32帧指标只作回归调试，不能据此选模型或推进 P4。
