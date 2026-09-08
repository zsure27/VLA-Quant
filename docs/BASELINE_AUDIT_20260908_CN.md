# 2026-09-08：基线审计、修复与执行入口

本说明取代旧文档中的 v2 profile / 自动 500 回合运行指引。仓库仍独立于 Triton；本次没有新增 GPU 模型结果，不能称为“QVLA 表格已复现”或“W2/A4 已修复成功”。

## 一、最重要的发现

### 1. OFT 的注意力后端不仅是性能选项

QVLA 的 `openvla-oft/pyproject.toml` 要求 `moojink/transformers-openvla-oft`，不是 PyPI 同版本的普通 Transformers。上游原依赖没有固定 commit。本轮固定为 `bc339d9ad707454c0c115970db43c260067c61ab`，并校验实际导入的 `modeling_llama.py` 内容指纹。

该 fork 的 `LlamaSdpaAttention.forward` 将掩码转为屏蔽 padding 的双向掩码，并使用 `is_causal=False`。但 `LlamaAttention.forward`（eager）仍直接使用因果掩码；SDPA 设置 `output_attentions=True` 还会回退到这个 eager 实现。**因此普通 Transformers、直接切 eager、直接打开 output_attentions，都可能改变 OFT 基线语义。**

修复：校准、教师、量化候选、评估全部固定专用 fork + SDPA；小模型运行时测试确认“改变后方 token 会影响前方 token”。可选 attention 探针从真实 SDPA 调用取得已经过 RoPE、GQA 展开、掩码转换的 Q/K，只旁路重建少量动作 query 的概率，原 SDPA 输出不变。

这不能证明旧服务器的 0/20 就是后端导致的：缺少当时实际加载源码的完整记录。新默认也不能直接继承旧 488/500 的成绩，必须重新测 BF16。

来源：[OFT 固定依赖声明](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/pyproject.toml)、[实际注意力实现](https://github.com/moojink/transformers-openvla-oft/blob/bc339d9ad707454c0c115970db43c260067c61ab/src/transformers/models/llama/modeling_llama.py)。

### 2. 量化范围对齐到什么程度

| 项目 | 本仓库新基线 | 确认程度 |
|---|---|---|
| 模型 | OpenVLA-OFT，Spatial checkpoint，双相机 + proprio，L1 连续动作头，8 步 action chunk | 固定模型配置；完整权重内容在运行时哈希 |
| DINO 分支 | patch Conv + 23 个动作连通 block 的 qkv/proj/fc1/fc2，共 93 个目标 | 既有连接性清单；新校准逐帧验证执行覆盖 |
| SigLIP 分支 | patch Conv + 26 个动作连通 block 的四类 Linear，共 105 个目标 | 同上 |
| LLM | 32 层 × Q/K/V/O/gate/up/down，共 224 个目标 | 精确模块名校验，不能只比较数量 |
| 保留 BF16 | projector、proprio projector、action head、embedding、norm 参数等 | norm 可等价重参数化，但不做低比特量化 |
| 不量化的算子 | QK/PV BMM、softmax、KV cache 等 | 明确列出，不能声称所有计算均为 INT4 |
| 校准数据 | 默认 8 帧用于工程诊断；可升至 32/128 等 | 不等于论文完整校准集 |
| QVLA 论文 SQ recipe | 模块级范围有依据；alpha、Conv 激活粒度、完整基线校准流程等未充分公开 | 未完全对齐，不能虚构默认值为论文值 |

论文明确量化视觉和语言、保留 projector/action head。开源权重量化选择器包含这两个 backbone 下的 Linear/Conv，并排除 lm_head。422 是该 OFT 动作连通路径的具体子集，不是所有 VLA 架构通用数字；更换架构必须重建 inventory，禁止硬套。

论文附录 F 的 512 指训练轨迹。现有提取器每条选一帧；即使来源覆盖 512 条不同轨迹，也不等于使用了论文的轨迹内部采样方式。正文还提及少量 instruction-only 数据，本仓库未伪造该子集。正式对比需要报告帧数、独立轨迹数、来源套件和采样方式。

来源：[QVLA 论文 §4.1 / 附录 F](https://arxiv.org/html/2602.03782v1)、[开源目标选择器](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/qvla/inject_fake_w.py)。

### 3. AWQ 修复与保留的适配边界

- 旧 W2 为整行对称、默认三电平；旧官方适配器的搜索为独立 Linear MSE。这两者不能作为完整官方 AWQ 的失败证据。
- 新正式入口支持 W2/W4/W8 A16；LLM **直接调用固定官方 `auto_scale_block`**，联合 QKV 以 attention 输出误差搜索，联合 gate/up 以非线性 MLP 输出误差搜索，并保留 V→O、up→down 的官方缩放结构。
- 每帧完整多模态 token 都进入 LLM 搜索。用短生命周期 attention 包装器逐帧回放真实 kwargs，再拼接输出计算官方损失；不让不同样本相互注意，不用文本代替图像/proprio，也不丢弃不同长度的指令。
- block 输入按 BF16 教师逐层传播；保存的 scale 在应用时真实修改 norm 和相关 Linear，再按官方 affine group 量化。不是给每个权重独立套 `Q(Ws)/s` 冒充 block 方法。
- 每层保存前先测**不量化的缩放等价性**，任一校准帧相对 MSE 超过 `1e-3` 则拒绝保存；这是 BF16 工程门槛，不是精度证明。
- clipping 调用官方函数，保留 Q/K 不裁剪规则；clip 损失采用 FP32 输入/权重降低校准溢出风险，属于显式数值适配。最终 Llama fake quant 保持 BF16 参数路径。
- 视觉分支仍为明确标记的 `vision_linear_adapter_v1`：官方 primitive + 自定义逐 Linear 搜索。官方 llm-awq 没有直接提供本模型双 ViT/patch Conv 的完整 recipe，**整个系统只能叫“官方 Llama block AWQ + 视觉适配”**。
- 视觉校准按“帧 × 两相机”给相同配额，逐帧校验调用次数；LLM block 不再只有稀疏均匀采样的 256 行。
- W2 必须独立搜索，不能加载 W4 的 scale/clip。group 默认 128；视觉 patch 权重列数不整除时补零再裁回，这也是显式适配。禁止把改变 group size 的实验隐去额外元数据成本。

来源：[官方 AWQ block 搜索](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)、[官方裁剪](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)。论文 Table 2 没有 AWQ W2；W2 是新扩展目标。

### 4. SmoothQuant：不掩盖 A4 的困难

使用固定官方 Norm→Linear smoothing、per-output-channel 对称权重量化、per-token absmax 激活量化，W/A4 是这些 primitive 的低位宽扩展；ViT 分组和 Conv 输入布局是本仓库适配，不能称为官方完整 VLA 支持。

422 个目标中，162 组 norm 平滑覆盖 258 个 Linear 输入，剩余 164 个输入仍可能做 A4。它们主要包括 attention 输出投影、MLP 第二投影以及 patch Conv；这不是凭数量就能断定的“漏平滑 bug”。A4 下这些位置的异常值、小通道归零更值得检验。

Conv 激活仍按 NHWC 的 RGB 通道做 per-token，而非 unfold 后 patch 的完整 K 维；先做 no-patch A4 消融，不擅自改变基线定义。所有 SQ 范围消融都维持同一套完整平滑组，仅改变 W/A 量化集合。

新增训练保护：现有 SQ hook 是 PTQ 推理用的 `detach + round`，若在需要梯度的路径调用则报错。否则未来训练 projector 时可能“代码正常跑，梯度却被 SQ 截断”。真正 QAT/STE 需要单独实现并通过反传测试，当前未冒称已经支持。

来源：[官方 SmoothQuant](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant)。

### 5. 种子不是只加一行 manual_seed

上游 `libero_utils.get_libero_env` 固定 `env.seed(0)`，注释明确指出即使用固定 init state，环境 seed 仍会影响物体位置。这是评估协议，不应直接删掉或替换为模型 seed。

本轮区分两个模式：

- `upstream`：保留环境 0、官方 init-state 顺序、不逐回合重新播种。用于原环境协议对照；不代表已恢复全部历史依赖/数值后端。
- `paired`：环境根种子和模型根种子分离，按 suite/task/init-state index 派生确定的回合种子，每回合显式设置；不受上一回合提前成功、异常或随机数消耗影响。BF16 和每个量化候选必须同模式同索引配对。

日志新增 `SEED_PROTOCOL` 和 `EPISODE_MANIFEST`，记录环境/模型 seed、初始状态内容哈希和索引；禁止超出 init-state 数量后取模重复计数。运行中设置 `PYTHONHASHSEED` 不会改变当前 Python 的哈希盐，因此在启动脚本设置。

训练方面：固定 OFT `vla-scripts/finetune.py` 未提供显式 seed 字段，也未调用完整播种函数，而数据来自 TensorFlow/RLDS。新增 `qvla/finetune_seeded.py`，在加载训练脚本前设置 Python/NumPy/PyTorch/TF，独立模型 seed 与数据 seed，开启确定性检查和 TF 数据顺序。只支持当前快速实验的单 GPU；未验证多卡分片，直接拒绝多卡。上游不保存完整数据游标/RNG，因此拒绝声称精确续训。TF 数据增强顺序、真实训练重复性仍需 AutoDL 验证。

另发现非 HF `PrismaticVLM` 构造器会按视觉维度重设 torch seed；它不在当前 HF OFT 入口的实际模型类中，本轮没有为了“清种子”修改这个未执行分支。上游 worker helper 的 LOCAL_RANK 也不能当全局 rank；当前 RLDS 单进程入口不依赖它。

来源：[环境种子](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/experiments/robot/libero/libero_utils.py)、[OFT 训练入口](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/vla-scripts/finetune.py)。

## 二、新服务器怎么跑

在新的目录 clone 本仓库。以下假设已经获得模型和 `sample-*.npz`，且样本目录有提取器生成的 `manifest.json`。旧服务器未恢复的数据不能从 GitHub 凭空找回，见 `data/README_CN.md`。

```bash
git clone https://github.com/zsure27/VLA-Quant.git
cd VLA-Quant
# 使用新的 ROOT，避免覆盖旧服务器上改动过的依赖。
ROOT=/root/autodl-tmp/qvla-repro INSTALL_ENV=1 bash scripts/bootstrap_autodl.sh

# 首轮只做 8 帧校准 + 8 帧轨迹级留出。打印的 OUT 要记下来。
STAGE=controls bash scripts/run_audit.sh

# 把下面路径替换成上一条实际输出，其他 CALIB/N/OFFSET/SEED 设置保持不变。
OUT=/root/autodl-tmp/qvla-repro/artifacts/baseline-audit-实际时间 STAGE=sq bash scripts/run_audit.sh
OUT=/root/autodl-tmp/qvla-repro/artifacts/baseline-audit-实际时间 STAGE=awq bash scripts/run_audit.sh
```

`controls` 先检查真实注意力实现、轨迹隔离、CPU 公式，再新校准 SQ，运行 BF16 教师、新进程 BF16 重复、只平滑 W16A16。默认重复动作 MSE 门槛 `1e-10`、平滑门槛 `1e-4`；不是普适理论界限，超限先回传日志，不能静默放宽。

`sq` 执行八种 W/A 与模态/oracle 分解；`awq` 分别重新校准 W4 与 W2，每种测 all/vision/language，再测 W2 projector oracle。默认不跑 500 回合，不启动训练，不打包整数内核。CPU/GPU 内存都可能受具体模型影响，先用默认少量帧；不要开多个 7B 进程抢同一块 GPU。

可选 attention：第一次 controls 时加入 `ATTENTION_LAYERS=7,15,23,31`，以后两个阶段保持相同值。每层只取最多 8 个真实 action-readout queries，保留全部 heads/keys；输出概率与模态质量，不改变 SDPA 后端。先验证有/无探针的 BF16 动作一致性，再解释 attention 图。

当前封装提供**源码与小模型层面的门槛**。完整 block 搜索、两相机校准调用、profile 重放、LIBERO 渲染和训练均未在本机实际 VLA 上跑通；任一失败应保留日志并修复，不得把报错删除后继续大批实验。

## 三、评估晋级与旧产物处理

Profile 已升级 v3，包含完整检查点文件哈希、校准帧内容哈希、官方五个核心文件与适配源码指纹、位宽、算法名、后端、种子和 block scales。旧 v2 仍可作为历史备份，但新正式入口拒绝加载；**不能只把版本号改成 3**。同一内容可迁移到另一目录，换权重必须重校准。

旧 `run-official-w4-smoke20.sh`、`run-official-quant-validation.sh`、`diagnostics/run_first_batch.sh` 已显式停用，避免复用旧 profile 或仅凭日志中的“500 episodes”跳过新的测试。

控制和离线诊断都通过后，先让 BF16 与候选各跑相同的 20 回合；只有 BF16 环境正常、候选有进一步验证价值时再跑 500。示例（先激活环境并设置 ROOT/OFT/PYTHONPATH）：

```bash
source scripts/activate_oft.sh
python "$ROOT/src/QVLA/openvla-oft/experiments/robot/libero/run_libero_eval.py" \
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --num_trials_per_task 2 --seed_protocol paired --seed 7 --env_seed 0 \
  --local_log_dir "$ROOT/eval/new-bf16-paired-7"

python qvla/run_eval_official_quant.py --method awq --weight-bits 2 --activation-bits 16 \
  --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --profile "$OUT/profiles/awq/w2.pt" --official-root "$ROOT/src/official-quantization" \
  --num_trials_per_task 2 --seed-protocol paired --seed 7 --env-seed 0 \
  --local_log_dir "$ROOT/eval/new-awq-w2-paired-7"
```

后续换根种子时三者（BF16、AWQ、SQ）全部配对重跑，分别报告每 seed 的结果。50 个 init states × 10 tasks 才是每种方法/seed 的 500 回合；不能把同一状态重复三遍混称更大独立样本。异常回合日志要单独分类，不把基础设施异常直接解释为量化精度退化。

## 四、验证记录

本地隔离 CPU 环境：Python 3.12、PyTorch 2.14.0+cpu、NumPy 1.26.4；官方 primitive 小测试通过；9 项回归测试通过（包括 attention 旁路不改变小模型前向）；固定 OFT fork 的双向 attention 测试通过；Python/JSON 仓库结构检查与 Bash 语法检查通过。目标 AutoDL 仍使用 QVLA 指定的 Python 3.10 / PyTorch 2.2.0，不能把本机新 CPU 版本测试称为目标环境已验证。

测试还实际捕获了一个路径污染问题：加入官方源码目录后，原 `qvla` namespace package 可能从旁边的历史目录加载旧评估器。新增显式 `qvla/__init__.py` 固定本包边界；旧 v2 和逐 Linear profile 的拒绝测试已覆盖这一回归。需要上游额外 qvla 工具时，在安装后的 OFT 目录运行，不通过临时拼接旧实验目录混用包。

仍待 GPU 的关键项：完整官方 AWQ block 搜索与重放等价性、实际 OFT BF16 基线、SQ 仅平滑误差、v3 profile 全量应用、校准与评估多 seed 配对、attention 旁路无扰动控制，以及 RLDS 真实数据训练重现。**当前状态是“审计后的候选基线”，不是已确认复现成功。**
