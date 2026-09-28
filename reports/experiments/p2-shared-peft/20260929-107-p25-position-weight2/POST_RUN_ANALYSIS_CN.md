# P2.5：位置权重2候选被离线门禁否决

同预算训练 exit0：exact12L blocks18–19 rank8 Response-SVD、peft_train80、1000步 AdamW 1e−4、Smooth-L1 beta0.1，唯一变化是归一化动作前三个位置维度权重1→2、按权重和归一化。训练参数1,249,280；初始/最终训练集 MSE `0.07039→0.01315`，adapter SHA256 `addc5bf81a44451c5d1acdfde2fc6811a3656b262822e0aab0c3cd09142215b4`。该训练误差不等于部署收益。

59条 `router_dev` 轨迹、295帧，旧模型与新模型的归一化动作 MSE 均值 `0.05173→0.05255`（略差），位置RMSE `0.18755→0.18652`（略好），夹爪符号分歧均为`0.06907`。相对12L的最差十分位轨迹增量 `+0.01681→+0.00907`，最大轨迹退步 `+0.04882→+0.01178`；两者均有31/59轨迹中位数优于12L。详细指标见同名 `results/.../router-dev-comparison.json`。这提示目标权重可压低极端退步，却没有整体稳定收益。

更关键的是在固定的task1/2各50个旧LoRA学生访问回合、query0–2共300个**完全相同观测**上：新模型对BF16 raw动作MSE均值为`0.07399`，旧LoRA为`0.01874`；新模型仅99/300低于旧模型。夹爪符号分歧从旧`0.01042`增至新`0.43042`，接近exact12L的`0.46458`。task1三个早期决策均是旧LoRA 50/50观测优于新模型，尤其query1夹爪分歧 `0.0025→0.9200`；task2 query0、2的位置误差确有改善，但query1夹爪分歧 `0.0025→0.3400`。详见同名 `results/.../paired-early-task-pos2-analysis.json`（包含逐任务、query与7维数据）。

**Gate：否决，未启动闭环。**位置权重2把task2一部分位置误差换成了task1大规模夹爪退步，是不合适的shared目标修正；不可凭最差十分位改善或训练损失下降声称恢复。也不继续调权重网格。下一项单变量是按P2.5主线改训练状态分布：保持旧uniform目标、参数/优化预算和peft_train Response-SVD初始化，用旧LoRA在指定训练reset访问的80个观测让冻结BF16同观测重标。训练reset0–4以后从闭环评价中全部排除，独立报告5–49口径，不触碰最终holdout。

本实验未产生新闭环成功率或新部署结论。原始训练、router_dev和300观测探针持久盘目录 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260929-107-p25-position-weight2/`；本机小指标归档同名 `backups/experiments/.../pos2-small.tar.gz`，SHA256 `10524eb0f99a7ff86418aa5e0bf763f2a9b4e9b6cb3fa7ba44d9e24db42bce89`。服务器和本机完整原始与runner状态归档 `pos2-full.tar.gz` 的SHA256均为`f6195841a4d2a3c2174afe475c8a42ed58baba920143ccddd2fa2e7eaa503085`。Git远端SHA在本轮总收尾记录。局限：训练外同观测误差不是反事实闭环，固定两任务是开发诊断，不能代替完整配对评测。
