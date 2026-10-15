# reset10–19门禁与累计150配对分析

三阶段全部exit0，PASS_B3_INITIALIZATION_PAIRED_PROTOCOL，初态/实际派生seed/首观测/实际命令与有限动作通过。本片Z=0/100，S=95/100；95救回、0新增失败，差95pp，任务经验95%区间[91,99]pp，精确McNemar p=5.0487e-29。

累计reset5–19，每侧150：Z=1/150（0.67%），S=144/150（96%）；143救回、0新增失败，差95.333pp，任务聚类经验95%区间[92,98.667]pp，精确McNemar p=1.7937e-43。十任务均观察到改善；微测重叠，不加入分母。

## 失败与解释

S累计六次失败：task6/reset7、task4/reset10、task6/reset16、task7/reset19、task9/reset10、task9/reset13，均两侧共同失败。Z仅task0/reset6成功，两侧均成功。其余143回合S救回Z。原始动作/观测及视频保留；不能从总成败断言夹爪、抓取或放置故障机制。Z因子全部真实更新，不是未训练零分支。

这支持当前A4、80早期训练观测、1000步固定设置下Response-SVD初始化比标准零输出强。S另用80独立校准轨迹，额外教师/初始化前向及分解成本须单列；并非等总信息预算，也不能外推所有标准LoRA。新S与历史B3不同，历史425/450不拼入本轮。所有样本为同十任务多次使用的开发初态，未读取holdout标签，不是盲测/未见任务。

## 成本与资源

本片策略查询Z2800、S1387；失败回合执行更长，查询数不同不能解释为模型算子加速。每侧adapter 19,988,480参数、BF16 tensor39,976,960字节，实际adapter文件40,133,618字节；底座/激活/优化器和SVD教师成本不包括在adapter字节中。两侧GPU总显存约16GB，墙钟含初始化、环境、trace、视频和归档干扰，不称真实INT2部署收益。

## 下一片

按预先顺序接reset20–29，每侧100；仅更改offset/count/output/lock，固定源码与已训练adapter不变。09:28:55UTC读到当前任务09:27:41UTC API响应，五小时剩30%、reset1792064715；磁盘11,312,607,232字节满足8GiB续片门禁。不是依据当前成绩方向选择。09:30:10UTC已启动supervisor21885，后续读实际进程。活动计划控制根B3-continuation-v1/reset-20-29/plan.json，文件SHA7ba43a6a179e9d0cf6bd02983884b725ccbc5927020b7d696a6da54c1038cc3c。reset10–19完整200视频增量正在生成，不能提前称双份完成。

## 完成分片的双份归档回执

reset10–19完整增量服务器`/root/autodl-tmp/qvla-repro/backups/046-b3-reset-10-19-complete-20261015-v1`，本机repo `.local/archives/20261015-046-quant-finetuning/`下同名目录。679,500,413字节，SHA256 `2cec7fcaf7059a61b18c8d25a6df97947801a61524fdc929b76a264a97e0da56`；4777项SHA/4778成员和200个原始视频逐项核验通过，保存全部轨迹/实际命令/退出码/门禁/358源码/150配对分析。活动reset20–29不包含。CN报告和最新本机回执另同步控制根，与冻结归档分开保存，不改原tar。

训练、micro/首50和本片的完整服务器/本机索引见同名results/COMPLETED_STAGE_BACKUP_INDEX_V1.json；原091来源归档及7B checkpoint保持原角色。阶段增量不是最终原生关机许可回执，最终仍须完整会话归档及服务器root短HEAD bundle。
