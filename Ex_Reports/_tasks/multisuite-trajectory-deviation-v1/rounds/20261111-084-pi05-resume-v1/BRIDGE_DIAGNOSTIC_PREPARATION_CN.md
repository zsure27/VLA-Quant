# π0.5后端数值桥接：CPU定位与下一步（未放行量化）

本轮原生参考采集使用新084身份；当前GPU计划及源码不改。下列检查不加载完整模型。

1. 已保存真实同query的两路预处理：三路图像（含掩码占位图）、图像mask和指令token/有效mask逐位相同。state存在约2.67e-8的表示差异；π0.5动作expert不直接加入state token，指令内状态token相同。该结果不等于全模型输入/输出等价，仍保留原探针FAIL。
2. 固定OpenPI的JAX Gemma将BF16 query先缩放，再以FP32结果计算QK；当前Transformers eager路径先把QK结果落到BF16再缩放。seed7、小张量CPU探针：原eager公式与JAX logits最大差0.00364756584，按JAX累积顺序的对照为0。这只支持一个可检验的算子精度假设，不证明它是完整动作RMSE失败原因。
3. 独立准备`pi05_attention_math_probe_v1.py`：仅用于下次独立登记的BF16同输入诊断，保持原模型权重、噪声、输入和门槛，不是新量化方法。不得未经GPU检查将其放进量化rollout。需在未量化模型中记录实际调用数、完整动作RMSE/max-abs、冻结及保存重载身份；若仍失败继续定位视觉/语言嵌入与逐层隐藏状态。
4. 历史v4 RMSE0.00256013347、v5 RMSE0.00279977320均超过0.002；最大绝对差门槛仍0.02。本轮没有放宽门槛，不能把跨框架误差作为量化漂移。原生参考只补初态5剩余19例，不以无量化配对的参考轨迹声称发现共性拐点。

后续方法目录仍保留十二项请求：BF16参考；GPTQ/AWQ/OmniQuant/ActQuant/QVLA W4A16；SmoothQuant/DuQuant/QuantVLA W4A8；SmoothQuant/DuQuant/QuantVLA/HoloQ-VLA W4A4。具体计数沿原目录（十二项量化配置，BF16另列）；名称不代表π0.5已接入。优先在桥接通过后处理具有可审计源码的GPTQ、AWQ、SQ8/SQ4、OmniQuant和DuQuant；专用VLA方法须核验π0.5架构适配，不用普通量化改名冒充。每个配置先4suite task0/初态5，必要门禁通过后再扩任务，复用原生参考与已完成病例。
