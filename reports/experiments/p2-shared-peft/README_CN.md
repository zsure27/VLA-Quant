# P2｜12L backbone 上的 shared PEFT

唯一默认底座为 [`awq-w2a16-12l-mixed-spatial-v1`](../../../configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)：DINO W2 G64、SigLIP W2 G128、语言 W4 blocks 8–15 与 20–23，其余语言 W2 G64，Spatial500 为 412/500。

第一版只在仍为 W2 的语言 blocks 18–19 上比较 shared Scale PEFT 与 rank8 Recovery LoRA，其他层精度保持不变。`20260917-107-response-svd` 只是零步初始化测量，不是已训练 PEFT 或运行时专家。

`20260925-107-p2-shared-peft` 已完成 exact-12L 的同预算离线训练。fixed-code Scale-PEFT
通过离线 gate；rank8 LoRA 的两种初始化改善连续动作误差，但夹爪分歧未改善。Scale 首个
配对开发闭环分片为44/50，12L与14L均为43/50；当前差值不足以通过稳定收益 gate，继续
使用同一 adapter、评测器和协议扩大闭环。P4专家和P5 Router仍未启动。
