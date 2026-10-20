# 035 修正环境条件下的首片 50 对硬门禁

门禁结论：`PASS_FIRST_SHARD_PROTOCOL`，允许下一片同协议 C0/C3 配对开发评测。微型 reset 5 各 10 回合与本片 reset 6–9 各 40 回合严格合并，**每配置 50 回合，无重复 reset**。C0 为 **42/50**，C3 为 **45/50**，rescue 4、break 1，净改善 **+3/50**。改善分别出现在 task 1、5、6，task 7 有退化；其余任务净差 0。按任务分层 bootstrap 的净成功率差 95% 区间为 **−2% 至 +14%**，精确 McNemar p=0.375，当前不支持“收益已稳定复现”的结论。

余下 40 对均运行 exit0，SHA 前后一致，命令除 C3 adapter 和输出目录外一致，40/40 首 query 观测在 C0/C3 间相同、40/40 首动作不同；全部策略查询 finite 且执行 8 步 chunk。逐回合配对由 evaluator manifest 核验；50 对合并后的 [`first50-gate.json`](../../../../results/experiments/p2-shared-peft/20260930-035-envseed1-first50-tail40/first50-gate.json) 和小结果在同名 `results`。完整含观测 `.npz` 的原始产物保留在服务器 `/root/autodl-tmp/qvla-repro/eval/a1-035-envseed1-first50-tail40-20260930/` 及本机 `backups/experiments/p2-shared-peft/20260930-035-envseed1-first50-tail40/`。

微型片与余下 40 对合计约 28 分钟运行时间（按服务器 runner 状态时间估算），只比较冻结 12L 的 C0/C3；14L/BF16 在此条件未配对测试，W4 恢复率为 `null`。条件是可观测的新环境随机流，官方初态索引 5–9 仍属历史开发范围，训练示范重叠审计未完成，不能称独立盲测。旧 110 的无效 A1 不计入样本量。

下一片预先固定 reset 10–19、10 任务×10 初态=100 回合/配置，继续保存 rescue/break 与逐任务翻转；不根据本片正差选择任务或改变回合范围。下一片完成仍需独立分析和备份，Router、额外 LoRA、扩大微调范围继续锁定，直到共享 C3 的闭环收益在充分独立条件上稳定。
