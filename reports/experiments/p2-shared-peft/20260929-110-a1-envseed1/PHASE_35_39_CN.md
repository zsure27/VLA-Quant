# 110 A1 环境种子复现：reset35–39 阶段分析

五个冻结配置各完成 50 回合，退出码均为 0；task/reset、model seed、env seed、初态哈希严格配对，无 episode error。此片仍使用原官方初态数组，只改变环境随机流，不能称为新初态或盲测。

| 配置 | 本片成功 | 累计 reset5–39 |
|---|---:|---:|
| BF16 | 50/50 | 339/350 |
| C0 exact-12L | 42/50 | 289/350 |
| C1 旧 demo-only rank8 LoRA | 42/50 | 284/350 |
| C2 14L 静态 W4 参照 | 44/50 | 306/350 |
| C3 student-state80 rank8 LoRA | 41/50 | 306/350 |

本片 C3−C0 为 **−1/50**，3 个救回、4 个新增失败，任务分层 bootstrap 95% 区间 −10 至 +8 个百分点，精确 McNemar p=1。任务 1 有 +1/5，但任务 3 和 6 分别为 −1/5、−2/5；收益在本片不稳定。C2−C0 仅 +2/50，恢复比例不能作有意义的效能结论。累计 C3−C0 仍为 +17/350，但 A1 支持门槛只能在预注册的 reset5–49 共 450 回合/配置完整后判定。

当前五小时实时额度剩余 79%。已核对最后一片 40–49 的本地和服务器计划 SHA 相同，GPU 空闲后启动服务器顺序执行器；继续固定模型、评测器、初态数组、动作语义和预算，不训练新 adapter、不启动 Router。原始完整产物保存在 `/root/autodl-tmp/qvla-repro/eval/a1-110-envseed1-35-39/`；Git 中配对小结果在同名 results 目录。SHA256：`paired-35-39-summary.json` 为 `aac5e62e1c929a69c270871b1fd0d36aeb14129abf4cea6f1ff0bfd91067a02f`，`paired-35-39-episodes.jsonl` 为 `e8df5113901c6669b0198a7b7d2e13088860ce62582e45d4dfc11bf5cacdfa1b`。最终完整归档和跨片失败分析在 450 回合结果齐备后补充。
