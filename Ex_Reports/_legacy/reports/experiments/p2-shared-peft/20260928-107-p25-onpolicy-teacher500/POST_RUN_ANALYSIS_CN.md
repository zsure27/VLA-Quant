# P2.5：500 条学生闭环轨迹的同观测 BF16 诊断

固定已完成的 data80 rank8 Recovery-LoRA 学生闭环轨迹，对全部 **7,642 次真实重推理观测**逐一向冻结 BF16 查询 8×7 动作及 H17。五片各 100 回合，成功数 83、83、77、83、82，总计 408/500，与原闭环记录一致；全部五片完成，`offline_final_holdout` 未触碰。所有样本来自学生当时已访问的观测，未来成败未作为策略输入。

按查询汇总，LoRA 与同观测 BF16 的 raw action MSE 均值 **0.03260**、中位数 **0.01773**，夹爪在实际 0.5 阈值后二值分歧 **9.17%**。先在每个 episode 内求均值，再分闭环结果：成功回合 MSE **0.02348**、失败回合 **0.04921**；失败减成功的 episode bootstrap 95% 区间约 **[0.0222, 0.0294]**。这只是关联，任务难度、轨迹长度与状态分布都可共同影响误差和成败，不能解读为动作 MSE 导致失败。

按 8 步 chunk 的查询级 RMSE，第 3 步约 0.1285，第 7 步约 0.1601；**352/500** 个回合的后段平均 MSE 高于前段，平均晚减早约 0.00606。夹爪连续误差维度较大（平均 RMSE 约 0.1842），但接近 0.5 决策阈值的预测比例很低：学生约 0.14%、教师约 0.22%。需区分数值误差、符号翻转和实际任务关键时刻，不把一处平均值当因果解释。

已有开发对照是 12L=411/500、LoRA=408/500、14L=431/500。完整闭环未恢复；本次较大样本诊断提示失败回合确有更高同状态教师分歧及 chunk 末端误差，但不足以单独确证 covariate shift。下一步已完成 peft_train 其他帧与 router_dev 的同口径补测，并预注册同一学生观测上 exact-12L 无 LoRA 对照，以检验 LoRA 相对简单底座是否真正降低这些状态的教师误差。P4/P5/Router 继续锁定。

小结果见同名 `results/experiments/p2-shared-peft/20260928-107-p25-onpolicy-teacher500/teacher500-analysis.json`。服务器原始归档在 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260928-107-p25-onpolicy-teacher500/teacher500-raw.tar.gz`，本机副本在 `C:\Users\zsure\Documents\Triton\.codex-work\vla-quant-sync-20260915\backups\experiments\p2-shared-peft\20260928-107-p25-onpolicy-teacher500\teacher500-raw.tar.gz`；两份 SHA256 均为 `04ce720badf00025034afb50ca37a9840dab85b945243b804caa067622867396`。原始归档保留逐查询 BF16 回答、对齐 JSONL、来源与完成标记；Git 仅提交分析小结果和恢复路径。
