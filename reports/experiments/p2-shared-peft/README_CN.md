# P2｜12L backbone 上的 shared PEFT

唯一默认底座为 [`awq-w2a16-12l-mixed-spatial-v1`](../../../configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)：DINO W2 G64、SigLIP W2 G128、语言 W4 blocks 8–15 与 20–23，其余语言 W2 G64，Spatial500 为 412/500。

第一版只在仍为 W2 的语言 blocks 18–19 上比较 shared Scale PEFT 与 rank8 Recovery LoRA，其他层精度保持不变。`20260917-107-response-svd` 只是零步初始化测量，不是已训练 PEFT 或运行时专家。
