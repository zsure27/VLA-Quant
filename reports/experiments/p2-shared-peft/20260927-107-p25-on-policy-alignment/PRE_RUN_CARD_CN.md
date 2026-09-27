# P2.5 同观测闭环对齐诊断：预运行卡

时间：2026-09-27。分类：DIAGNOSTIC。当前主阶段：P2.5；锁定阶段：P4、P5、P6。

## 来源与研究状态

依据 `docs/VLA_RESEARCH_GOVERNOR_20260926.txt`、`docs/P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md`、2026-09-26 data80 稳定性报告及原始配对结果。data80 LoRA 42/50，exact-12L 43/50，14L 43/50。该 slice 的 W4 headroom 为 0，恢复率为 null。离线 data80 同时改变轨迹数和优化步数，只能归为 confounded development evidence。Spatial500 中 411/412 与 430/431 的来源审计尚未结束，当前不据此声称精确 canonical 成功率。

## 实验准入

- PRIMARY STAGE：P2.5 闭环对齐诊断。
- QUESTION：rank8 Recovery-LoRA 在自身动作诱导的观测上，是否仍与冻结 BF16 教师一致？
- HYPOTHESIS：历史固定观测改善未转化为闭环收益，可能由学生访问分布上的动作分歧增加导致，尤其是 8 步 chunk 后段和夹爪阈值附近。
- BACKBONE HASH：运行前记录 exact-12L profile、checkpoint、data80 LoRA 和评测器的 SHA256；本卡不以未完成的 500 回合漂移审计替代这些哈希。
- METHOD：在历史开发 reset 0–4 上每任务 1 回合，共 10 个学生 episode；保存每次真实 re-inference 观测、学生 8×7 动作和 H17，然后冻结 BF16 在同一序列化观测上查询。无训练。
- DATA SPLIT：历史 Spatial 开发初态；仅机制诊断，不作为无偏最终测试。`offline_final_holdout` 封存。
- ONE CHANGED VARIABLE：评测时加入观测与 H17 记录；学生模型、动作执行和评测初态不变。随后教师对同一已保存观测查询，不与学生同时驻留 GPU。
- CONTROL：相同 data80 检查点、exact-12L 精度配置、blocks 18–19、rank8、seed=0、10 个 reset、8 步完整 open-loop chunk。固定观测 peft_train/router_dev 的现有数据仅作为分布参照，未完成轨迹级 router_dev 前不做正式三分布推断。
- PRIMARY METRIC：按 episode 聚合的同观测学生与 BF16 教师 8×7 动作分歧及随 query 位置变化；连续值和夹爪阈值分歧共同报告。
- SECONDARY METRICS：逐 step/逐维位置与旋转误差、gripper margin、首次夹爪分歧、H17 cosine、成功率（诊断性，不作门禁）。大分歧阈值保持 null，待 router_dev 预注册。
- EXPECTED DECISION：若分歧随学生访问显著增加且与失败对应，进入同 rank8、同预算 student-state teacher relabeling 的预注册控制；若没有，先查损失与 8 步执行位置的关系及动作语义，不能直接扩 rank、专家或 Router。
- STOP DECISION：10 回合及同观测教师查询后停止；若代码/语义校验失败则停止并修复同一诊断，不扩大搜索。单次 GPU 时间上限 2 小时，超时保留中间产物并分析。
- OUTPUT DIRECTORY：服务器持久盘 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260927-107-p25-on-policy-alignment`；本仓库同名 reports/results，完整本机副本归入 backups。
- ESTIMATED COST：学生 10 回合加教师查询，预计低于本轮五小时额度主要部分；在阶段边界复查实时五小时额度。

该实验无论支持或否定假设，都会改变下一步诊断或训练决定。不得将其成功率标为 final，也不借此越过 P4/P5 门禁。
