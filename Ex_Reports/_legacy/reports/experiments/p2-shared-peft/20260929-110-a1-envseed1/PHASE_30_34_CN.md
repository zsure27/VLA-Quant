# 110 A1 环境种子复现：reset30–34 阶段分析

五种冻结配置各完成 50 回合，退出码均为 0，task/reset、模型与环境 seed、初态哈希严格配对，无 episode error。该片仍沿用官方初态数组，只改变可复现的环境随机流，不能称为新初态或盲测。

| 配置 | 本片成功 | 累计 reset5–34 |
|---|---:|---:|
| BF16 | 49/50 | 289/300 |
| C0 exact-12L | 40/50 | 247/300 |
| C1 旧 demo-only rank8 LoRA | 41/50 | 242/300 |
| C2 14L 静态 W4 参照 | 45/50 | 262/300 |
| C3 student-state80 rank8 LoRA | 47/50 | 265/300 |

本片 C3−C0 为 +7/50，8 个救回和 1 个新增失败；任务分层 bootstrap 95% 区间为 +6 至 +22 个百分点，精确 McNemar p=0.039。增益主要在任务 1（C0 0/5、C3 4/5），任务 5 也改善 2/5；任务 6 从 5/5 降到 4/5。C2−C0 为 +5/50；小样本的比值估计不作为“超过 W4”的证据。累计 C3−C0 为 +18/300，C2−C0 为 +15/300，最终 A1 判定仍需完成 reset35–49。

已登记的 35–39 计划与服务器副本 SHA 一致，五配置顺序执行器在五小时额度刷新后启动。后续仍固定模型、backbone、评测器及动作语义，不训练新 adapter，也不启动 Router。原始完整产物在服务器 `/root/autodl-tmp/qvla-repro/eval/a1-110-envseed1-30-34/`；Git 中配对小结果在同名 results 目录。结果 SHA256：`paired-30-34-summary.json` 为 `687714bd36b28870ab742dd257019fc87cc3cd90a0957e0c138bc469c78226ea`，`paired-30-34-episodes.jsonl` 为 `54f44087e505504d58887cd57d7f53eb080a6cd556ba3f844555c042e6e055dd`。

前三片（reset5–29）已形成服务器持久盘归档 `/root/autodl-tmp/qvla-repro/backups/a1-110-envseed1-core-20260929-192600-852d350/`，复制到本机被 Git 忽略的 `backups/experiments/p2-shared-peft/20260929-110-a1-envseed1/` 同名目录，10 个归档条目的 SHA256 全部一致。30–34 及后续分片会在最终收尾时补入完整归档。当前阶段结果说明 C3 有正向复现趋势，但独立初态和跨 seed 鲁棒性仍未得到证明。
