# VLA AWQ W2A16 / SmoothQuant W4A4：源码审计与下一轮诊断

日期：2026-09-02。对象：当前 Triton 工作区中保存的 OpenVLA-OFT 代码、旧 W2 实验和 official-source W4 适配器。此报告没有新的 GPU 成功率结果，也没有覆盖/修复现有基线代码。

## 1. 先纠正结论和实验命名

你的论文判断是对的。QVLA Table 1 中 OpenVLA-OFT + SmoothQuant W4A4 的 Spatial 为 77.2%，四套任务平均为 73.4%。此前本地 0/20 不能简单解释为“SmoothQuant 天生不能做 W4A4”。但论文未给出足以逐项对齐我们 SQ 适配器的完整基线 recipe，不能把这个差距直接等同于某一行代码的错误。[论文 Table 1](https://arxiv.org/html/2602.03782v1#S4.T1)

Table 2 的 AWQ 只有 W8A16、W4A16，没有 AWQ W2A16 结果。W2 是更激进的新目标，不能由论文 W4 结果推出它应该成功。论文报告的 LIBERO 实验 GPU 是 RTX 4090；不要用 A100/4090 差异解释当前崩溃。[论文 Table 2 与设置](https://arxiv.org/html/2602.03782v1#S4.T2)

历史日志中的结果（用户此前提供，本轮未重跑）：

| 路线 | 结果 | 正确口径 |
|---|---:|---|
| BF16 Spatial | 488/500，97.6% | 浮点基线 |
| 新适配器 AWQ W4A16 | 491/500，98.2% | 官方底层函数 + 自定义逐 Linear 搜索的 fake quant |
| SQ W4A16 | 19/20 | 小样本控制，不是完整成功率估计 |
| SQ W4A8 | 20/20 | 同上 |
| SQ W8A8 | 20/20 | 同上 |
| SQ W4A4 | 0/20 | 明显需要审计的 A4 退化 |

此前把新版称为“完整官方 AWQ”不够准确，应该纠正。所有现有路线仍是 BF16 执行的伪量化，不能把耗时和模型占用当成 packed INT2/INT4 的性能。

## 2. 官方源码有没有被改坏？

已在线获取不可变 commit，并逐文件对比本地文件，忽略 CRLF/LF 换行差异，以下五个文件内容全部一致：

| 官方仓库 commit | 已核对文件 |
|---|---|
| llm-awq `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | `awq/quantize/quantizer.py`、`auto_scale.py`、`auto_clip.py` |
| smoothquant `c61476d728e42ae0d8a35e7e78494edcac3237b5` | `smoothquant/smooth.py`、`fake_quant.py` |

这是源码文件内容比对，不是所有依赖或整个仓库的完整审计，也不能自动证明 AutoDL 当时导入的就是这些文件。原加载器只调整 sys.path，未验证已缓存模块实际来源；新增探针会检查真正导入的函数路径和 SHA256。

参考：[官方 AWQ 搜索实现](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)、[官方量化器](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)、[官方 SmoothQuant](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/fake_quant.py)。

## 3. 已确认的实现偏差

### A. 旧 W2 不是官方 AWQ，且默认只有三个有效码值

定位：`qvla_experiments/calibrate_scaled_w2.py:60`，以及导入它的 sequential/action-aware 分支。

旧实现按整个输出行做对称 absmax 量化，W2 的 qmax=1。虽然 clamp 范围写成 [-2,1]，默认 clip_ratio 为 0.8～1 时，量化前的归一化值范围最多是 [-1.25,1.25]，round 后到不了 -2，只使用 {-1,0,1}。因此默认网格全都是三电平，而不是官方 group-wise affine 的四码量化。

同时存在三处算法差异：

- 无 group128，整行共享 scale，更易被少量大权重决定分辨率。
- 激活统计用 absmax，而官方 AWQ 搜索用 mean(abs(x))。
- 旧搜索还包含自定义 weight-stat 分母，且候选/目标都是逐层重构；并非官方搜索流程。

这不表示“三电平实现数学上非法”，而是它不能被标为官方 AWQ W2。旧失败只证明旧配置失败。新版 official-source 入口又只允许 W4，所以目前并没有完成一个可审计的完整官方流程 W2 基线。

### B. 新版 AWQ 调用了官方 primitive，但重写了关键搜索目标

定位：`openvla-official-quant-adapter/qvla/official_quant_adapter.py:170`。

新版采用官方 affine group128 量化、20 点 scale 搜索形式和官方裁剪。但自定义 `search_awq_scale` 对每个 Linear 独立优化，默认只取 256 个输出行。

官方 Llama 的 `auto_scale_block` 会联合处理 Q/K/V，并用 attention 子模块输出重构来选 scale；gate/up 联合搜索时看 MLP 非线性后的输出。新版只最小化局部 Linear MSE，不能感知 QK → softmax、SwiGLU、残差和动作端放大。因此“每层局部 MSE 很小，但策略失效”完全可能。

注意两个不要误判的点：

- 新旧代码都有 `Q(Ws)/s` 形式；只要量化发生在中间，这不是无效的尺度抵消。它是有效的 dense fake-weight 表示，但不是已打包的低比特矩阵。
- 官方当前 AWQ 搜索本身也用激活均值的幂，不要误称新版缺少“官方必须的 weight-scale 分母”。

### C. 视觉校准存在真实的覆盖遗漏

旧 sequential：`calibrate_scaled_w2_sequential.py:91` 在目标层第一次调用后立即抛出 StopAtTarget。当前模型按主相机、腕部相机顺序调用同一视觉编码器，所以视觉目标只捕获主相机，遗漏第二次调用。action-aware 分支完整前向，没有同样的提前退出问题。

新版：`calibrate_official_quant.py:72` 默认 256 行 / 32 样本 = 每次调用 8 行；视觉层每样本调用两次，前 16 张样本就耗尽 256 行预算；LLM 每样本调用一次，才覆盖到 32 张。SQ absmax 的统计不受这个行数上限影响，仍会统计所有调用。

修正方向是按“样本 × 相机 × token 类型”分配预算，而不是简单提高总行数。LLM 还要显式覆盖视觉、文字、proprio、动作读出位置，不能只均匀抽取长序列的少数位置。

### D. Profile 的一致性检查不足

定位：`run_eval_official_quant.py:46`。原加载器检查方法、版本、重复 target，但最终使用第一份 shard 的元数据，没有严格验证每一份的 checkpoint、group_size、alpha、样本清单与源码 hash 一致。恢复脚本的“已有 profile 可用”检查也偏弱。

另有 FP16 落盘：AWQ input_scale/clip_max、SQ activation_absmax 会存为 FP16。超过 65504 可溢出，微小值会发生舍入/下溢。这是需要测的风险，不是已经观察到非有限值。新的探针会拒绝非有限 profile、混合元数据、错误 target/shape，以及变化的 SQ 源码。

## 4. SmoothQuant W4A4：哪里最值得查？

在真实 422 个 target 名单上静态执行分组函数，得到：

| 部分 | 数量 | 说明 |
|---|---:|---|
| Norm → Linear 平滑组 | 162 | 参数会重参数化，但仍用 BF16 存储和计算 |
| 被平滑的 Linear 输入 | 258 | QKV、gate/up、视觉 fc1 |
| 未被平滑但仍做 A4 的输入 | 164 | 视觉 attn.proj/fc2 共 98，LLM o/down 共 64，patch conv 共 2 |

不能把这 164 个直接称为“漏写”：官方 SQ 对 Llama 本来就主要处理 Norm 后 QKV 和 gate/up。但从 W8A8 下探到 W4A4 时，它们可能成为瓶颈，尤其 `down_proj` 输入经历 SwiGLU 后产生的通道不均衡。

per-token A4 的量化步长约为 max(abs(x))/7。某通道幅值低于该 token 最大值的约 1/14 时，可能直接舍入为零；A8 的相应门槛约为 1/254。这使模态相关的小信号在 A4 更容易丢失。需要看 zero fraction、max/RMS 和局部 A 量化误差，而不只看全张量直方图。

另外，Conv 激活适配是 BCHW→BHWC 后在 RGB 的 C=3 维做 per-token 量化，即每像素 3 通道一组，并非 unfold 后每个 patch 的完整 K 维分组。官方 SQ 不直接提供这一视觉扩展，论文 SQ baseline 的 Conv 细节也未对齐。这是定义差异，不能先验认定改成 patch 分组一定更好。最便宜的定位实验是 patch 输入保持 A16，其余仍 A4。

已排除的几个常见误读：

- 适配器在调用原地 SQ 量化函数前 clone 了输入，没有直接污染共享 residual 输入。
- 官方函数中未接住 `t.view(...)` 的返回值，不会改变后续沿最后一维求最大值的语义，不能据此认定核心公式错误。
- 当前没有把 QK/PV 的 BMM 强制做 INT4；官方 Llama fake-quant 默认也不量化 BMM 输入输出。这不是遗漏一行就能解决的问题，必须明确 W4A4 的算子范围。
- 本地校准 helper 和评估 runner 都没有显式强制 FlashAttention2。不能在未读服务器 manifest 前认定两者 attention backend 已经不一致。

## 5. 如何判断 attention、模态与 projector 的因果关系

### 5.1 先拆 W 与 A，再拆视觉与语言

固定 checkpoint、预处理、输入帧、proprio、指令、seed、attention backend。先做 BF16、仅平滑 W16A16、W4A16、W16A4、W4A8、W4A4。仅平滑应接近 BF16，允许 BF16 重参数化的舍入误差，但若误差已接近完整 W4A4，应先停止精度调参检查实现。

接着比较 vision-only、language-only。所有 SQ 消融保持同一套完整 smoothing 分组，只改变 W/A 量化的 target，避免“少量化了几层”同时改变平滑定义。

不要预设 attention 最严重。attention-only 与 MLP-only A4 都值得跑；如果 down_proj 的局部噪声更大、保留该处 A16 的动作恢复更明显，就应优先 MLP。不同集合参数量不同，不能只按总 MSE 给模块做“固有敏感度”排名；再做逐 block 的一次扰动/一次恢复比较。

### 5.2 同一输入，沿深度看误差在哪里出现

分开看：两种视觉编码器 × 两个相机视角 → vision 拼接输出 → projector 输出 → LLM 输入 → 每一层 hidden states → action head。

OFT 当前走连续 L1 action head。这里的“token 分布”主要是 token 位置上的连续特征分布，不应把任意 hidden channels 做 softmax 再报告 KL，并称之为语言 token 概率。应记录：相对 MSE、cosine、RMS、通道/位置误差、零值比例；动作端报告归一化 7 维、物理 7 维、chunk 各步、夹爪离散分歧。

本探针按模型实际 action-head 读出切片记录 action_readout，而不是武断地把最后 56 个位置当作动作。视觉特征图按主/腕相机调用分别记录；LLM 按主图像、腕部图像、文字、proprio、动作读出位置分组。图是抽样 token、完整 channel 的误差，不能当全 token 精确均值。

### 5.3 用 oracle 接口替换判断 projector 微调是否有希望

同一个量化模型、同一个输入，只把 projector 输出替换成缓存的 BF16 teacher projector 输出，其余量化 LLM 保持不变。

- vision-only 量化 + oracle 后接近 BF16：证明这个接口替换链路有效。
- 全模型量化 + oracle 后动作明显恢复：视觉到语言接口漂移值得优先修复。
- 全模型 + oracle 仍差，language-only 也差：LLM 量化本身是硬瓶颈，仅调 projector 大概率不够。
- projector 特征 MSE 很大但动作不坏：部分特征误差落在动作不敏感方向，不能只凭特征图调参。

oracle 是因果诊断，不是可部署模型，更不是“projector 微调一定能达到的性能上限”。它向量化模型提供了量化视觉中可能已经不可恢复的信息。

如果以上证据支持 projector：先冻结视觉/LLM，只在校准训练轨迹上训练 projector（或小残差/低秩修正），监督量化视觉输入对应的 teacher projector 特征，并加入归一化 action distillation。只匹配均值和方差可能不够，还要保留方向/相关结构。按轨迹划分 train/held-out，再用 rollout 验证。不要用 LIBERO 正式评估初始状态训练。

### 5.4 真正的 attention 图放在第二轮

优先检查 action-readout queries 指向主图像、腕图像、文字、proprio 的注意力质量总和、entropy、teacher/student attention JS divergence、top-k key overlap。只在已经定位的少数层/头采样，避免保存全部注意力矩阵。

如果需要切到 eager 才能取 attention，teacher 和 candidate 必须一起切换，先测 BF16 原 backend→eager 的控制误差。不能从未处理 RoPE 的 q_proj/k_proj 输出直接画“真实 attention”：还需正确的 RoPE、KV head 扩展、mask 和缩放。本次脚本不输出 attention 矩阵，以免产生这一类误导。

## 6. 现在在 AutoDL 跑什么

### 6.1 文件与运行

把整个 `diagnostics/vla_quant_audit` 文件夹上传为：

`/root/autodl-tmp/qvla-repro/diagnostics/vla_quant_audit`

保留原环境和源码，不要替换现有 qvla 目录。脚本使用此前已有的 qvla helper/adapter、官方源码、SQ profiles、模型和 sample-*.npz。默认 8 个探针样本，offset=64，避开旧 SQ 校准前 32 个文件；这只是帧级留出，不保证轨迹级独立，也不用于发布成功率。

```bash
bash /root/autodl-tmp/qvla-repro/diagnostics/vla_quant_audit/run_first_batch.sh
```

需要持久会话时，在你已有的 screen/tmux 会话中运行即可。默认输出是独立时间戳目录，不覆盖任何旧结果；每个 case 重新加载一个模型进程，不会在已量化权重上再次量化。

默认路径可用 ROOT/OFT/SAMPLES/OFFICIAL/PROFILE/CKPT/TARGETS 环境变量覆盖。少量样本调试可用 `N=2`；正式第一轮仍建议默认 8。若路径或 profile 验证失败，把日志发回，不要绕过验证或新旧 shard 混用。

### 6.2 第一轮内容

1. CPU 小测试：旧 W2 码值、官方 affine W2、平滑等价性、输入 clone、平滑 scope。
2. AWQ 主线微探针：8 个代表目标，前 4 帧局部校准、后 4 帧局部验证；比较旧 symmetric-row RTN W2、官方 affine-group128 RTN W2/W4、新适配器 Linear-search W2/W4。微探针重新搜索 W2，不复用 W4 scale/clip。
3. BF16 teacher 缓存。
4. SQ 控制组及 vision/language 分离、projector oracle 两种控制，共 9 个 SQ case。
5. 生成真实测量的 LLM 深度误差图、视觉分支/相机深度误差图、7 维动作误差图。

第一轮不跑完整 rollout、不训练 projector、不进行完整 422 层 W2 搜索。AWQ 微探针没有调用官方完整 block 搜索，名称明确标注为 adapter_linear_search，结果只说明局部量化器/搜索差异，不是最终策略质量。

输出要点：

- `awq-micro/awq_micro.json`：held-out 局部输出误差。
- 各 case 的 `metrics.json`：特征误差、局部激活量化误差与 zero fraction、动作 7 维和 chunk 各步。
- `manifest.json`、`scope.json`：实际源码/预处理/样本 hash、attention backend、W/A target。
- `sample0_feature_deltas.npz`：首样本 projector、LLM 输入/末层的 token×channel 差值，供后续画共享色标热图。
- `figures/*.png`、日志。
- `teacher/*.pt` 是本机复用缓存，不需要传回；只读取自己生成的可信 .pt 文件。

源码与 sample 内容做 hash；模型大权重没有全量 hash，运行期间不要修改 checkpoint。探针关闭原 helper 自动改写本地 checkpoint 配置/模型源码的动作，并记录实际加载模型的源码 hash。若 plot 阶段提示缺少 matplotlib，可在该既有实验环境中安装后单独执行 plot_results.py，不用重跑模型。

### 6.3 看到第一轮结果后，再选最少的第二轮

设置 OUT 为第一轮输出目录后，可以复用同一 teacher：

```bash
ROOT=/root/autodl-tmp/qvla-repro
OFT=$ROOT/src/QVLA/openvla-oft
DIAG=$ROOT/diagnostics/vla_quant_audit
source "$ROOT/envs/qvla-oft/bin/activate"
source "$ROOT/env.sh"
export PYTHONPATH="$OFT:$ROOT/src/LIBERO:${PYTHONPATH:-}"
# 替换成终端打印的真实第一轮路径：
OUT=/root/autodl-tmp/qvla-repro/artifacts/vla-quant-audit-YYYYMMDD-HHMMSS
common=(--checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
        --samples-dir "$ROOT/calib/action-space-balanced/libero-512"
        --official-root "$ROOT/src/official-quantization"
        --targets-file "$DIAG/qvla-connected-422.txt"
        --teacher-dir "$OUT/teacher"
        --profile-dir "$ROOT/qvla/spatial/official-w4-vl/smoothquant"
        --mode smoothquant --num-samples 8 --offset 64)
python "$DIAG/probe.py" "${common[@]}" --activation-scope attention --output "$OUT/a4-attention"
python "$DIAG/probe.py" "${common[@]}" --activation-scope mlp --output "$OUT/a4-mlp"
python "$DIAG/probe.py" "${common[@]}" --activation-scope smoothed --output "$OUT/a4-smoothed"
python "$DIAG/probe.py" "${common[@]}" --activation-scope unsmoothed --output "$OUT/a4-unsmoothed"
python "$DIAG/probe.py" "${common[@]}" --activation-scope no-patch --output "$OUT/a4-no-patch"
python "$DIAG/probe.py" "${common[@]}" --activation-math-fp32 --output "$OUT/a4-fp32-math"
python "$DIAG/plot_results.py" --root "$OUT"
```

以上第二轮 W 都保持 422 层 W4，仅 A4 范围不同。不要一次混入 alpha、group size、截断 percentile 和 projector 训练。A4-fp32-math 只改变激活 fake-quant 的计算精度，不改变其位宽/粒度；它是数值排错控制，不是免费获得更高精度的正式结果。

## 7. AWQ 主线后续修正顺序

1. 用微探针确认官方 affine-group W2 相比旧三电平的改善/退化发生在哪里。
2. 独立建立 W2 profile 格式/入口，记录 bits=2、group_size、完整源码与样本 manifest；禁止加载 W4 clip 到 W2。
3. LLM 使用官方 `auto_scale_block` 的共享 QKV、共享 gate/up 及子模块重构目标；传入真实 VLA embeddings、position IDs、mask 等，不能退化成纯文本校准。先在 1 个 Llama block 测搜索前后不量化的函数等价性和实际 action error。
4. 视觉没有可直接套用的官方 OpenVLA AWQ block recipe，应明确标注视觉适配；qkv 联合关注 attention 输出，fc1/fc2 关注 MLP 输出，保留两种编码器和两个视角。
5. W2 下单独消融 group128→64→32、clip on/off、更多校准 token、动作读出位置加权，逐个变量比较。group size 减小增加 scale/zero-point 元数据，不能只报“都是 2 bit”。
6. 先做 held-out 动作误差和 20 episode 冒烟，再做 500 episode；如果 uniform W2 无法恢复，可将敏感层保留 W4 作为额外 mixed-bit 对照，但必须报告平均位宽，不能继续叫纯 W2。

原官方 AWQ pre_quant 也会在 scale/clip 之前计算下一 block 的输入，不能把“不是逐层传播量化噪声”本身当作偏离官方的证据。用量化输入作校准可另设实验，但应独立命名。

## 8. 本轮验证边界

已完成：五个官方源文件在线内容比对；真实 target 名单的分组计数；旧 W2 默认网格的数值码值检查；新增 Python 语法编译和 Bash 语法检查。

未完成：本机没有 PyTorch/GPU 模型环境，self_test 的张量数值测试、模型探针与出图需在 AutoDL 执行。没有声称新增脚本已通过真实模型运行，也没有声称某一假设已经解释 0/20。原有代码、profile 和 checkpoint 均未修改。

第一轮最值得回传的是：self-test.log、awq_micro.json、smooth-only/W16A4/W4A4/vision-only/language-only/oracle 的 metrics.json 和 figures。用这些结果决定是修量化校准、处理 A4 outlier、保护特定算子，还是启动 projector 微调。
