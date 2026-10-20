# P2.5：peft_train、router_dev 与学生访问状态的描述性比较

本轮使用 P1 v3 轨迹切分。新补测的 peft_train 取 80 条不同训练轨迹的 20% 时位帧，与原 80 个优化校准文件不同；固定 BF16、exact-12L、同一 data80 rank8 LoRA，在同观测上比较。router_dev 使用既有 59 轨迹 × 5 时位＝295 帧；student-visited 为已核验的 500 条 LoRA 闭环轨迹、7,642 次查询。未训练新模型，`offline_final_holdout` 未触碰。

| 分布 | 样本与权重 | LoRA raw action MSE 均值 | LoRA 夹爪二值分歧 | 无 LoRA MSE 均值 |
|---|---|---:|---:|---:|
| peft_train 其他帧 | 80 轨迹，各 1 帧 | 0.02125 | 3.75% | 0.02049 |
| router_dev | 59 轨迹，各 5 帧 | 0.02206 | 6.91% | 0.02258 |
| 学生访问 | 500 回合，7,642 次查询 | 0.03260 | 9.17% | **待同状态对照** |

在 peft_train 其他帧，LoRA 虽在 76.25% 的帧上降低 MSE，但少数较大退化使均值比无 LoRA 高约 0.00076；router_dev 上均值低约 0.00052、57.63% 帧改善，收益很小。学生访问查询的 LoRA MSE 均值比两个离线分布高，失败回合误差也更高。这是**分布错配的线索，不是确认**：三组的轨迹、时位、采样权重与任务阶段不同，学生访问数据还由 LoRA 自身生成。当前不能据此直接扩大数据或开启 student-state 重标训练。

已预注册并运行更直接的控制：从每个 LoRA 闭环回合固定取第 4 次因果查询（执行 24 步后），在这 500 个完全相同的学生状态上查询 exact-12L 无 LoRA，并核验 BF16 输出。若 LoRA 在自身访问状态仍优于无 LoRA 而闭环净损失，则优先检查动作目标/夹爪关键决策和 8 步执行；若相对收益主要在学生访问状态消失，再考虑同 rank8、同预算的教师重标。专家与 Router 继续锁定。

逐样本和轨迹清单见同名 `results/experiments/p2-shared-peft/20260928-107-p25-distribution-triad/`。原始帧、教师及双模型输出保留在服务器持久盘 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260928-107-p25-distribution-triad/peft-train-control-raw.tar.gz`，本机副本在 `C:\Users\zsure\Documents\Triton\.codex-work\vla-quant-sync-20260915\backups\experiments\p2-shared-peft\20260928-107-p25-distribution-triad\peft-train-control-raw.tar.gz`；双份 SHA256 均为 `6cda793187275307427fd14337145b7f51cd332d4a6cce9016507b60c6011930`。
