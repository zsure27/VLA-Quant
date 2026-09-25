# P1 数据与参数化契约报告（107，2026-09-25）

本轮完成了 PEFT/Contextual Routing 主线的 P1 门槛。训练数据按完整轨迹切分，固定码
Scale-PEFT 在真实 blocks 18–19 上复现官方 W2/G64 fake quant，Recovery LoRA 的冻结、
零输出、梯度和重载约束也通过。

关键数字：432 条轨迹、52,970 个 transition；`298/59/75` 对应
`peft_train/router_dev/offline_final_holdout`。固定码覆盖 14 个 Linear，零残差最大误差为
0，可训练 scale residual 共 6,324,224 个。完整逐层哈希和轨迹角色见配对的
[`results` 目录](../../../../results/experiments/p1-data-contract/20260925-107-p1-data-contract/README_CN.md)。

P1 通过只说明数据和参数化实现可信，并不说明 Scale-PEFT 或 LoRA 能恢复闭环成功率。
后续 P2 必须先在同一 12L 底座、同一 blocks 18–19 上证明 shared 方法的留出收益。
