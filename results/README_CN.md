# 结果数据说明

`official_quant_validation.csv` 是根据历史实验日志中的最终统计恢复的结构化摘要，不是本轮重新运行产生的结果。原始逐 episode 日志不在本地，不能从该 CSV 反推出任务级成功序列。

AWQ 的 500 episodes 与 BF16 500 episodes 可以用于“未观察到明显精度损失”的判断，但置信区间重叠，不能宣称 AWQ 显著优于 BF16。SmoothQuant 各项只有 20 episodes，仅用于定位配置是否完全失效。
