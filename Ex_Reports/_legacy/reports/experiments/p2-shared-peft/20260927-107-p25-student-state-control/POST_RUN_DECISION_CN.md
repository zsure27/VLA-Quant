# P2.5 学生访问状态配对控制：阶段决定

状态：完成，机制诊断，P4/P5/Router 继续锁定。140 个学生实际访问观测及源文件 SHA 全部核验，exact-12L 无 LoRA 与已有 data80 rank8 LoRA 比较；唯一模型变化为是否加载 blocks18–19 的 LoRA。新 probe 的冻结 BF16 动作与原同观测 BF16 在全部 140×8×7 个数值上完全一致（最大绝对差 0）。`offline_final_holdout` 未触碰。

以冻结 BF16 为参照，LoRA 的逐 query 动作 MSE 均值为 0.02855，无 LoRA 为 0.03528；配对差为 -0.00673。LoRA 在 50% 的 query、7/10 个 episode 上更好。按 episode 先求均值，配对差为 -0.00543；10 episode bootstrap 95% 区间约 [-0.01374,+0.00318]，跨零。夹爪符号分歧 LoRA 为 4.64%，无 LoRA 为 15.71%。逐帧原始指标及配对摘要见同名 `results` 目录。

结论：LoRA 在其自身访问状态上降低平均动作误差，尤其减少夹爪符号冲突；但改善分布不均，episode 级 MSE 不确定性仍跨零。此前同一开发 slice 的闭环为 LoRA 42/50、12L 43/50、14L 43/50。离线动作改善未转化为该闭环 slice 的成功率改善，且该 slice 无 12L→14L headroom，W4 恢复率必须为 null。不能据此证明 covariate shift 是唯一原因或 student-state 蒸馏必然有效。

下一轮优先在固定 exact-12L、blocks18–19、rank8 条件下做更严格的失败回合与夹爪时位分析，再预注册 student-state relabeling 与 offline-only 同预算对照；必须保留跨 seed、锁定有 headroom 的闭环 slice。当前额度进入收尾区，不再扩展 GPU 实验。尚未完成 411/412 与 430/431 来源审计，因此全局基线继续标为约 411/500、约 430/500，不混用精确值。
