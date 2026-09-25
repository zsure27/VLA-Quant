# SQ baseline源码审计（CPU/文献，不是新GPU结果）

本轮主GPU测试为BF16/AWQ W4配对分片。以下仅代码和公开源审计，不能记作SQ性能修复完成。

|项目|已核验当前实现|尚需验证|
|---|---|---|
|W原语|官方quantize_weight_per_channel_absmax；我们在FP32权重副本操作，再cast回BF16|QVLA表格baseline的实际dtype和量化器|
|A原语|官方per-token absmax；Linear最后维每token；Conv NHWC每像素通道|作者对Conv patch/token的实际定义及包含目标|
|4bit码域|qmax=7，理论对称−7..7共15码值；浮点舍入误差需要实测码域|作者是否用了affine16码值；15码本身不证明bug|
|注意力|仅Linear/Conv输入量化；QK/PV等BMM输入未单独量化|作者实际开关；官方Llama quantize_bmm_input默认False|
|平滑|Llama RMSNorm→QKV与gate/up；视觉LayerNorm→QKV或fc1|完整分支配对与bias、共享输入统计、尺度坐标|
|数值|局部FP32组近等价；先前BF16及FP32配对端到端有动作扰动|舍入累计与实现错误区分；完整视觉FP32支路尚未测|

官方源码来源：[fake_quant.py](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py)、[smooth.py](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py)。本项目实际运行还严格校验固定官方源SHA，不以当前main页面替代固定版本。

作者仓库树只读审计见qvla_public_tree_audit.json：tree SHA26cc4821a3be4c003d09d3c7997b38db2a347982，返回未截断，路径中未匹配smoothquant/awq/baseline/fake_quant关键词。**这是文件名检索，不能证明没有其他命名的实现或未发布的实验代码。** 论文表格的alpha、实际W/A原语、范围及校准配置仍列未确认；不能选择更容易的设置后声称严格复现。

下一步按照实现审计→完整视觉FP32成对数值控制→W16A16/W4A16/W16A4/W4A8/W8A8/W4A4固定因素对照推进。max动作MSE1e-4只是既有工程筛查阈值，需同时参考代数等价、同入口重复误差及配对闭环，不作单独根因判据。
