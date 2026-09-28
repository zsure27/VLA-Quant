# P2.5 exact-12L Recovery-LoRA 配对 Spatial500：结果与门禁

状态：五个预注册分片全部退出 0；每配置 500 个历史/开发 reset，10 任务 × 50 初态。三配置在同一评测器、seed 和初态清单上配对，未触碰 `offline_final_holdout`。这是开发证据，不能称为新的盲测。

| 配置 | 成功 | 与 12L 差值 |
|---|---:|---:|
| C0：exact-12L | 411/500（82.2%） | — |
| C1：12L + data80 rank8 Recovery-LoRA | 408/500（81.6%） | −3/500（−0.6 个百分点） |
| C2：14L 静态 W4 参照 | 431/500（86.2%） | +20/500（+4.0 个百分点） |

C1 与 C0 共同成功 383、仅 C0 成功 28、仅 C1 成功 25、共同失败 64；McNemar 精确检验 p≈0.784。按回合配对 bootstrap，C1−C0 的 95% 区间为 −3.4 至 +2.4 个百分点，跨零。由于本轮 C2−C0=20/500>0，按预注册口径的点估计 W4 恢复比例为 −15%；其 bootstrap 区间约 −142% 至 +50%，波动很大，不能解读为精确恢复能力估计。C1 与 C2 相差 23/500，配对仅 C1 成功 13、仅 C2 成功 36，p≈0.0014。

更具体地，在 27 个“C0 失败而 C2 成功”的回合中，C1 救回 18 个，仍失败 9 个。但 C1 也在其他位置引入 28 个相对 C0 的失败，抵消了救回。任务差异显著：任务 1 为 C0/C1/C2=1/11/9，任务 2 为 47/33/50，任务 5 为 26/34/34，任务 7 为 45/40/47。这样的正负交错提示需要分析目标、状态分布与 8 步 chunk 的控制语义；仅凭任务差异不能宣布可路由专家互补，更不能用任务标签或未来成败训练 Router。

## 与历史结果的关系

本轮同评测器 C0=411，C2=431。此前静态 12L/14L 报 412/431，Scale-PEFT 同协议对照报 411/430。至少存在一回合来源漂移；本报告只使用本轮三配置严格配对的 411/408/431，不混合历史分母。历史漂移的 evaluator、profile、checkpoint 与 init manifest 哈希仍需逐项审计，不能凭总数相近推定原因。

## 成本、失败与下一步

LoRA 仅作用于 blocks 18–19，rank8，adapter 状态 2,509,154 字节，SHA256 `ce34247710923b1ff2db9f40c5a0b982cdd228a5238d19b4dd9005e4743c9a47`；评测器 SHA256 `0889196c6a9863fc586f53ecc0831fa46afb3952d576c099e0f6bca765759299`，分片脚本 SHA256 `2c2b76585e600a1a4110098258dd7753d4038d8d48ad35dc66797b66ffab9e20`。五片共执行 1,500 次闭环 rollout，从 05:39 至 10:04 UTC 约 4 小时 25 分；脚本运行时长包含模型加载与环境初始化。离线动作 MSE 和同状态夹爪分歧曾改善，但此次完整闭环没有净恢复，**shared PEFT gate 未通过**。

下一项为已预注册的 P2.5 同观测诊断：对全部 C1 学生访问观测由冻结 BF16 教师查询，比较逐 step、位置/旋转/夹爪、早晚查询和失败时序，并与 peft_train/router_dev 指标对齐。只有证据支持 covariate shift，才做同 rank8、同优化预算的学生状态教师重标对照。P4/P5/Router、静态 W4 搜索与 holdout 继续锁定。

## 数据与核验

- 小型配对结果：同名 `results/experiments/p2-shared-peft/20260928-107-p25-paired-headroom/paired-spatial500-summary.json`、`paired-analysis.json`；其中原聚合器沿用历史字段名 `scale_minus_12l_95ci`，本轮实际代表 C1 LoRA 减 C0，本文按正确语义表述。
- 服务器持久盘原始归档：`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260928-107-p25-paired-headroom/paired-spatial500-raw.tar.gz`。
- 本机原始副本：`C:\Users\zsure\Documents\Triton\.codex-work\vla-quant-sync-20260915\backups\experiments\p2-shared-peft\20260928-107-p25-paired-headroom\paired-spatial500-raw.tar.gz`。
- 双份原始归档 SHA256：`7e7f3fd429d2dbc1134ed4bc04a8d9d54dee1e9f1859ea451b7d3198fd97dc21`，归档 7,791 个条目；聚合摘要 SHA256：`194507f677e73d6bd6b33252190b67222a1c017b85b06bd43c48eabc7aacd676`。
