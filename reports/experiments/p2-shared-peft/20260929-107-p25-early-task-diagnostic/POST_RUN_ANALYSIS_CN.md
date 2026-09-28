# P2.5：两个任务前三次决策的同观测结果

五个自动接续分片均 exit0，固定task1/2各50个开发回合、每回合query0–2，共300个观测；逐样本检查序列和观测SHA。冻结BF16、exact12L、静态14L及历史LoRA的8×7动作，在LoRA学生实际访问的**相同观测**比较。原始小指标服务器与本机同名归档 `backups/experiments/p2-shared-peft/20260929-107-p25-early-task-diagnostic/early-small.tar.gz`，SHA256 `ddb5fa9792ad6e6693c8fb376aefeb462e537c57ae0da3fc24f70e634194516f`；逐任务、query、7维结果和来源哈希见同名 `results/.../paired-early-task-analysis.json`。服务器完整原始目录为 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260929-107-p25-early-task-diagnostic/`。

| 任务/决策 | LoRA MSE | exact12L MSE | 静态14L MSE | LoRA/12L夹爪分歧 |
|---|---:|---:|---:|---:|
| task1 query0 | 0.01683 | 0.07375 | 0.05781 | 0.0025 / 0.4125 |
| task1 query1 | 0.01097 | 0.17006 | 0.15114 | 0.0025 / 0.9675 |
| task1 query2 | 0.02392 | 0.15244 | 0.12347 | 0.0500 / 0.8875 |
| task2 query0 | 0.02196 | 0.00933 | 0.01333 | 0 / 0.0250 |
| task2 query1 | 0.02060 | 0.07921 | 0.07416 | 0.0025 / 0.4900 |
| task2 query2 | 0.01818 | 0.00369 | 0.00324 | 0.0050 / 0.0050 |

task1 是“碗在 ramekin 旁”，其12L夹爪方向与BF16在早期严重不一致，LoRA在每个时间点全部50/50观测上降低总动作MSE，同时几乎消除夹爪分歧。task2是“碗在桌面中央”；LoRA在query1也强烈修复夹爪，但query0仅5/50、query2仅4/50比12L低MSE。task2的LoRA第一个位置维度 MSE 在query0约 `0.1146` 对12L `0.0341`，query2约 `0.0447` 对12L `0.0061`。第4次决策此前还观察到task2 LoRA MSE `0.1375` 对12L `0.0361`。由此可见，局部夹爪修复与位置漂移同时存在，不能单凭总平均MSE或夹爪改善判定闭环收益。

**决策：**不全局替换14L教师，也不启动Router/专家。预注册一次同数据、同1000步、同rank8和初始化的shared LoRA单变量训练：仅将归一化动作前三个位置维度Smooth-L1权重从1调为2，并按权重和归一化。先比较轨迹级router_dev和学生访问观测的task2位置、task1夹爪与worst-decile；只有双向门禁通过才进入配对闭环。该权重是单点机制测试，不扫超参。

局限：任务因历史差异而被选为开发诊断，不能当无偏评测门禁；这些状态由旧LoRA访问，其他模型没有真实反事实轨迹。位置误差是否造成闭环失败尚无独立时序/接触标注。按成败分层是事后分析，不进入训练或策略输入。旧模型的闭环仍为411/500、408/500、431/500；本诊断未新增闭环成功率。完整原始数据及runner状态已在服务器与本机同名 `backups/experiments/.../early-full.tar.gz` 归档并核对SHA256 `eaa2ce096cf7ede295d4673baf5f1cf1f216f65f974478cfbefc296b5cbe6a28`。Git提交及远端SHA在本轮同步后记录。
