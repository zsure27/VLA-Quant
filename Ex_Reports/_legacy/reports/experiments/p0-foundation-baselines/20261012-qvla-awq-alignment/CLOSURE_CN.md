# 091 无卡审计与B5前置收尾

## 结果和边界

已完成无卡条件下可执行的内容审计及前置：实际W4 profile、22项checkpoint内容身份、422目标名称/形状与分散权重行的整数码重建通过；21项CPU契约通过。发现实际旧OFT部署仍采用会覆盖环境种子的调用次序，已在独立overlay修正并固定SHA，旧产物与来源文件保留。

QVLA Spatial BF16/W4为97.6%/93.0%，本项目历史500为97.4%/97.2%。92.5%是论文四suite平均。W4的4.2pp差距仍待实测归因：论文AWQ完整配方未公开，不能调差基线追分或声称严格复现。真实环境/processor探针被2GiB无卡环境系统Killed，CPU已通过不等于真实输入或GPU门禁通过。

B5代码、训练/运行入口和最终v2不可变计划已注册，来源Git `eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d`，GPU未启动、正式router未训练。用户下次开卡后先真实输入探针、BF16/A0桥接、bank/static query10，再采集标注和训练router，最后微测/首50审计及450开发评测；不能直接长跑。[执行卡](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/docs/091_B5_NEXT_GPU_GATE_20261012_CN.md)

备份后、删除已验证的重复导出tar，持久盘实测9,716,518,912字节，约9.0GiB，低于12GiB首次起跑门槛；需扩容或事先审阅可再生缓存。原始实验和SVD spool未删除。本轮提前收尾原因是无卡可行工作结束、真实门禁需用户开卡和足够RAM/磁盘，不是额度耗尽。

## 双份备份

- 本轮服务器：`/root/autodl-tmp/qvla-repro/backups/cpu-audit-091-final-20261012-v1`。
- 本轮本机：[完整归档](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/backups/experiments/p0-foundation-baselines/20261012-qvla-awq-alignment/cpu-audit-091-final-20261012-v1)。包含profile、B3/B4训练材料、student-state80、代码/完整Git bundle、最终计划和审计回执。服务器与本机均核验3＋141项SHA；143项初始导出包SHA `017d34f5d0cd6adfc1acf89bc28d1f5024d023102e25d75c54010c6a170dfbe2`，CPU关机helper作为双方已核验增量。
- 旧046、091完整归档已恢复到本机标准P2忽略目录并分别核验13＋18,494、8,952＋8,897项；完整导出包双方SHA `8399ed0ff145c1162cf3eca1c9414aa184f9cfe7e08417bef323000e40c78668`。公共7B模型仍在持久盘，22项文件SHA和恢复路径记录在本轮审计中，未把模型权重提交Git。

## 原生关机

标准GPU helper的dry-run失败：无卡启动的`/usr/bin/nvidia-smi`是零字节占位文件，shell退出0不能证明GPU空闲。独立CPU v2 helper验证实际无NVIDIA设备节点、CUDA不可用/设备数0、无活跃实验以及完整归档和唯一原生supervisor后执行。历史GPU及旧无卡helper字节保留。

北京时间2026-10-12 17:46:12，收到`execute=true`回执，supervisor751；随后SSH由远端关闭，退出-1。**平台OFF及停止计费未独立核验。** 本轮心跳已PAUSED，Windows未关闭。回执保存于本机完整归档及[结果JSON](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/results/experiments/p0-foundation-baselines/20261012-qvla-awq-alignment/native-shutdown-receipt.json)。执行源码Git与服务器旧root Git852d350分别记录，不混为同一运行版本。
