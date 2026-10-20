# router_dev 轨迹级门禁：数据契约

分类：CONTRACT。当前主阶段 P2.5。问题：能否从冻结的 v3 轨迹分割中仅提取 router_dev 的固定时间位置帧，建立用于 A/B/C/D 因子比较的可复现开发集？

本次唯一操作是按每轨迹归一化位置 0.1/0.3/0.5/0.7/0.9 抽帧；角色固定 router_dev，预期 59 条轨迹、最多 295 帧。控制为 `/root/autodl-tmp/qvla-repro/backups/experiments/p1-data-contract/20260925-107-p1-data-contract/trajectory-inventory-v3/trajectory_split.json`，SHA256 `3560425653ef4b297a48a3854aae20348898129b6673db1474943e95897606a2`，数据集 `libero_spatial_no_noops/1.0.0`。程序逐轨迹校验稳定ID并保存轨迹/帧 SHA。peft_train 仅用于训练/Response-SVD 校准，offline_final_holdout 封存；本次不抽取后者。

成功时仅将该集合提升为开发门禁，随后固定 teacher、exact-12L、A/B/C/D 的同样本评估，并报告 frame 与 trajectory 两种聚合、改善轨迹比例、最差十分位、逐维/夹爪。失败时停止下游评估并修复数据契约。历史 32 帧仍只用于回归。当前不做最终评估，亦不改变 backbone 或 P4/P5 gate。
