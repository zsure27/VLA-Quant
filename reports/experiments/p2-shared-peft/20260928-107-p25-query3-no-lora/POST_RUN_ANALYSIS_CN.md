# P2.5 同学生访问状态无 LoRA 对照：运行后分析

状态：五片全部退出 0，合计 500 个已预注册的 `query_in_episode=3` 学生访问观测，每回合约已执行 24 步。样本与此前冻结 LoRA 策略的 BF16 同观测查询逐条按 `observation_sha256` 配对；500 条教师夹爪 signed margin 与历史记录逐值完全一致。其他六个教师动作维度尚未做逐值重算核验，这是本次契约验证的局限。数据仅来自 Spatial states0–49 的历史/开发回合，未触碰 `offline_final_holdout`。

| 同状态配置 | BF16 raw action MSE 均值 | 夹爪符号分歧率 |
|---|---:|---:|
| exact-12L，无 LoRA | 0.047196 | 23.0% |
| exact-12L + data80 rank8 Recovery-LoRA | 0.033024 | 6.575% |

LoRA−无 LoRA 的配对 MSE 均值为 −0.014172，回合配对 bootstrap 95% 区间约 [−0.02068, −0.00687]；500 个观测中 421 个 LoRA 更接近教师、79 个相反。按十个任务分层，七个任务平均改善、三个任务平均退步，存在明显任务异质性。逐维比较显示 LoRA 对夹爪、第一位置维和旋转维有收益，但第二、第三位置维的均方误差反而增加，不能用总体 MSE 掩盖此差异。按原 LoRA 闭环结局分层，成功回合对应 query3 平均改善 −0.01535，失败回合为 −0.00893；结局是事后标签，仅用于诊断，不能作为策略输入或因果效应。

同期严格配对 Spatial500 是 exact-12L 411/500、LoRA 408/500、14L 431/500。因此 LoRA 在**自己访问的相同状态**明显更接近 BF16，但未带来净闭环收益。这削弱了“LoRA 一到学生状态便完全失去动作拟合”的简单解释；更应检查动作目标与闭环关键步骤的对齐、任务异质性以及 8 步 chunk 中的错误累积。此探针没有让无 LoRA 独立 rollout，不能推断若换成无 LoRA 后这些回合会成功或失败。现有数据仍不足以确认 covariate shift 是主要因果机制，不能直接启动学生状态教师重标训练。

**门禁判断：** shared PEFT 的稳定闭环恢复仍未成立；P4/P5/Router、静态 W4 搜索、rank/SVD/Scale 网格继续锁定。后续优先在已有配对轨迹上定位 LoRA 造成的 28 个新增失败与 25 个救回回合的最早关键分歧，按任务、步骤、夹爪与位置维记录，提出一次只改变一个主变量的 P2.5 目标修正；训练前须使用训练 reset 重新采集，不能将本次评测状态当训练样本。当前没有经预注册且能在剩余额度内完成的下一项 GPU 阶段，完成双份归档与 GitHub 同步后关闭 107，避免空跑。

## 产物与成本

- 配对聚合、小结果与任务分层：同名 `results/experiments/p2-shared-peft/20260928-107-p25-query3-no-lora/paired-query3-analysis.json`；原始小结果归档 SHA256 `f9d1bb942d7c67f865d9552782dda260a3b0e594d2d10a26dcb4deb872b3f943`。
- 五片各 100 个教师/无 LoRA 模型查询，共 500 个同状态探针；模型是冻结模型，没有训练、参数增加或新的闭环 rollout。LoRA adapter 2,509,154 字节，SHA256 `ce34247710923b1ff2db9f40c5a0b982cdd228a5238d19b4dd9005e4743c9a47`；完整耗时与磁盘成本以原始 manifest/运行日志为准。
- 连续执行器 revision 17，五阶段退出 0，计划 SHA256 `1fcb84e299b4865199e8724d3def33616da2ae9460050c4434c3638bf923f128`。
- 服务器持久盘：`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260928-107-p25-query3-no-lora/query3-control-raw.tar`，SHA256 `5fc9a70545f9f9201d77599a2661697b0e4f6b1b9fa540cdf87a0dafbedad94d`。本机同名完整副本传输及哈希核验见收尾记录；大体积原始观测和模型探针不会提交普通 Git。
- Github 仅保存小结果、预注册卡与本报告。最终提交和远端 SHA 见收尾记录。

## 收尾核验（2026-09-28）

完整原始归档在服务器持久盘和本机各一份，均为 2,168,606,720 字节、1,072 个 tar 条目，SHA256 均为 `5fc9a70545f9f9201d77599a2661697b0e4f6b1b9fa540cdf87a0dafbedad94d`。本机路径：`C:\Users\zsure\Documents\Triton\.codex-work\vla-quant-sync-20260915\backups\experiments\p2-shared-peft\20260928-107-p25-query3-no-lora\query3-control-raw.tar`。五个先前的配对闭环与教师诊断归档仍各有服务器持久盘和本机副本，SHA 见各自同名报告。本轮 GitHub 小结果和分析已推送至 `zsure27/VLA-Quant` 的 `codex/repository-cleanup-20260922` 分支，远端 SHA `89be7f270c9fda42c3188b7c46316234d34e586e`。原始 2.1GB tar 未上传 GitHub；关机执行回执另存。
