# C模块｜SmoothQuant与激活量化恢复

2026-10-13覆盖：用户要求研究重心转向VLA量化微调，开发C模块，并将相同LoRA恢复方法扩展到激活量化；旧“Router有效后才做SQ”的排序已被覆盖。当前是研究设计，未启动GPU；091已结束、心跳PAUSED，下一轮用户自行开机告知。

完整计划：[量化微调研究计划](../p2-shared-peft/20261013-quant-finetuning-research-plan/REPORT_CN.md)。设计身份：[C-SQ-v1登记](../../../configs/model_registry_c_smoothquant_v1.json)。

C0–C6依次W16A16（只平滑）、W8A16、W8A8、W4A8、W4A4、W2A4、W2A2。W4A16/W16A8/W16A4为独立交叉控制。C5/C6为条件性探索；优先C0数值等价、W/A误差拆分，再C3/C4 PTQ/零输出LoRA/Response-SVD对照。原9-29 PEFT C0–C3保留，新增身份均为C-SQ-v1命名空间。

先审计平滑的norm/fan-out/视觉/卷积等价，不能让LoRA补偿未通过等价门禁的实现。新SQ底座需重新训练适配器，量化激活输入与STE梯度路径须明确；BF16输入旁路须独立命名和计量。W/A目标范围、保护模块、真实字节/显存/延迟与伪量化限制均单列。