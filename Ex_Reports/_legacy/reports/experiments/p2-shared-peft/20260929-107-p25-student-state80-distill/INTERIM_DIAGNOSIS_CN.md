# P2.5：学生访问状态重标的离线门禁结果

训练与两组离线探针均exit0；闭环450回合正在按照同初态计划执行，故本文不报告其成功率。训练侧固定10个任务各4个reset（0–3），每回合query0、1，共80个原始学生访问观测，SHA校验后由冻结BF16在同一观测生成教师动作。Response-SVD仍只用原peft_train80；损失回到uniform Smooth-L1 beta0.1，1000步、rank8 blocks18–19、AdamW 1e−4，1,249,280个可训练参数。训练样本和模型摘要见同名 `results/.../student-state80-manifest.json`、`training-sample-hashes.json`、`train-summary.json`。新adapter SHA256 `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`；训练集最终MSE `0.000776`仅说明拟合，不是闭环指标。

**独立59条router_dev轨迹、295帧**：新/旧data80模型的归一化动作MSE均值 `0.04915/0.05173`，位置RMSE `0.18278/0.18755`，夹爪符号分歧 `0.06102/0.06907`，优于12L的轨迹数为 `36/59` 对 `31/59`。最差十分位相对12L的轨迹增量 `+0.01213` 对 `+0.01681`；但中位帧MSE新模型 `0.03490` 高于旧 `0.03308`，不把均值改善夸大为所有轨迹改善。详见同名 `results/.../router-dev-comparison.json`。

**训练外reset5–49的固定query3同观测诊断**：在旧LoRA策略产生的450个观测上，新/旧raw 8×7动作相对BF16 MSE `0.01915/0.03260`，配对差 `−0.01346`，按回合bootstrap 95%区间约 `[−0.02045,−0.00729]`；夹爪符号分歧 `0.06056/0.06528`。新模型仅逐点优于旧模型190/450，表明它主要修正较大误差。task2从 `0.14030→0.02494`，task1从 `0.03419→0.01334`；task4、8、9均值反而升高，需闭环检验。任务及维度明细见同名 `results/.../paired-query3-450-analysis.json`。原冻结12L与14L在同450状态上的MSE `0.04736/0.04113`；不可把这一诊断当反事实rollout。

**门禁决定：**进入同初态450回合闭环，而不是进入P4/P5/Router。旧控制在完全相同reset5–49为12L `368/450`、旧LoRA `366/450`、14L `388/450`，静态headroom20。闭环新模型与旧控制的seed、初态哈希及评价器/profile SHA逐回合核验；失败时停止配对解释。训练reset0–4一律不评价，不能和原Spatial500拼合或宣称最终盲测。

原始训练及离线结果服务器持久盘在 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/`。服务器与本机同名完整归档 `backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/student-state-full.tar.gz` SHA256 `ad1252f1e7c9a62ad5ebdd1a26ae6106aff88ae77d804c70673364b6995166f3`（本机复核后确认）；小指标归档 `student-state-small.tar.gz` SHA256 `14296f7616b87213511aecd042d778b2ca741845b507cc7b623654be142b2916`。Git提交与远端SHA在本轮总收尾更新。
