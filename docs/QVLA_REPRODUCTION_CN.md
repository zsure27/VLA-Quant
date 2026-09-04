# QVLA 完整复现与验证指南

## 1. 先定义“复现成功”

本项目不把“脚本可以运行”当作论文复现。证据分为四级：

| 级别 | 要回答的问题 | 当前官方代码状态 |
|---|---|---|
| R1 基线 | BF16 OpenVLA-OFT 能否达到论文成功率？ | 可复现 |
| R2 发布实现 | 官方 Hessian proxy + fake weight quant 的行为是否稳定？ | 可验证 |
| R3 论文算法 | 动作空间敏感度与 `{0,2,4,8,16}` 分配是否能重建？ | 需自行实现 |
| R4 部署性能 | 真实 W4A4 是否达到 4.5GB、1.49x？ | 需真实低比特内核 |

最终报告必须分别写 R1-R4，不能把 fake quant 的成功率和理论位宽换算成真实显存/速度。

## 2. 新手需要掌握的最小知识

### 2.1 VLA 闭环

输入是图像、语言指令和机器人状态，输出是动作。动作改变环境，下一帧又依赖刚才的动作，因此很小的单步误差也可能在长时序中累积。分类模型只需比较一次标签，VLA 必须同时检查单步动作误差和最终任务成功率。

### 2.2 PTQ、fake quant 与 real quant

- PTQ：对训练好的模型用少量校准数据确定量化参数，不重新训练完整模型。
- Fake quant：权重先量化再反量化，GEMM 仍使用 BF16/FP16。它验证精度损失，不证明真实显存和加速。
- Real quant：权重以 packed INT4 保存，并由能消费 INT4 的 CUDA/Triton kernel 计算，才可验证物理显存和速度。

### 2.3 QVLA 的 W4

QVLA 的 W4 是目标模块的平均通道位宽约为 4，不是所有通道固定 4 bit。每个输出通道从 0、2、4、8、16 bit 中选择；0 bit 表示剪枝。Projector、Action Head 和 LM Head 保留 BF16。

## 3. AutoDL 目录和省钱规则

推荐单卡 RTX 4090 24GB，至少 8 vCPU/32GB RAM，数据盘总容量 200GB 起步。

```bash
mkdir -p /root/autodl-tmp/qvla-repro/{src,models,data,outputs,logs,artifacts}
cd /root/autodl-tmp/qvla-repro
```

规则：安装和下载时也按开机计费；长下载完成后立即保存环境指纹。按量阶段先跑 1 trial，再跑 10 trials，确认无误后才跑 50 trials。不要在排错时启动四套完整评估。

## 4. 获取代码并冻结版本

```bash
cd /root/autodl-tmp/qvla-repro/src
git clone https://github.com/AutoLab-SAI-SJTU/QVLA.git
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C QVLA rev-parse HEAD
git -C LIBERO rev-parse HEAD
```

把两个 commit SHA 写入每次实验记录。后续不要无记录地执行 `git pull`。

## 5. 创建主环境

4090 上优先使用 CUDA 12.1 版 PyTorch 2.2.0。宿主机驱动可以更新，但 Python 包版本需固定。

```bash
conda create -n qvla-repro python=3.10.14 -y
conda activate qvla-repro
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

cd /root/autodl-tmp/qvla-repro/src/QVLA/openvla-oft
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

pip install -e /root/autodl-tmp/qvla-repro/src/LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

先验证环境，不要急着跑模型：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
import transformers, timm, tokenizers
print(transformers.__version__, timm.__version__, tokenizers.__version__)
PY
```

预期关键版本：Python 3.10、Torch 2.2.0、torchvision 0.17.0、timm 0.9.10、tokenizers 0.19.1、FlashAttention 2.5.5，以及仓库指定的 Transformers fork。

将本工作区的 `scripts/` 和 `configs/` 上传或同步到服务器项目根目录，然后执行：

```bash
bash scripts/capture_env.sh artifacts/environment-initial
```

## 6. R1：先复现 BF16 基线

### 6.1 Smoke test

只跑 LIBERO-Spatial，每个任务一次，共 10 episodes：

```bash
cd /root/autodl-tmp/qvla-repro/src/QVLA/openvla-oft
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --center_crop True \
  --seed 7 \
  --local_log_dir /root/autodl-tmp/qvla-repro/logs/bf16-smoke
```

验收：模型正常下载；MuJoCo 能创建环境；10 episodes 完成；日志包含 `Final results`；没有 NaN、维度错误或持续 OOM。Smoke test 的成功率没有统计解释，不与论文 97.6% 比较。

### 6.2 Pilot test

把 `--num_trials_per_task` 改为 10，共 100 episodes。记录总耗时，然后估计完整 500 episodes 的费用：

```text
预计完整时间约为 pilot 时间 x 5
预计费用 = 预计小时数 x 当前 4090 每小时价格
```

### 6.3 Paper-scale test

每套 10 个任务、每任务 50 episodes，即每套 500 次。四个 checkpoint 对应四套任务。套件、checkpoint 和论文目标见 `configs/paper_targets.json`。

完整跑完后解析日志：

```bash
python scripts/summarize_libero_log.py logs/bf16-spatial/EVAL-*.txt \
  --json-out outputs/bf16-spatial-seed7.json
```

初次验收建议：单套结果落在论文值的约 2 个百分点内；出现更大偏差先排查版本、GPU、center crop、checkpoint、初始状态和 action chunk。正式结论报告成功数/总数和 95% Wilson 区间，不只报告百分比。

## 7. R2：验证官方发布的 fake-quant 实现

### 7.1 校准输入

论文报告从 LIBERO 训练数据随机采样 512 条轨迹；发布脚本实际只读取如下 JSONL：

```json
{"image": "/absolute/path/to/frame.png", "text": "pick up the black bowl"}
```

它没有读取动作标签，因此这一步验证的是发布的 Hessian/input-covariance proxy，不是论文描述的 action-space sensitivity。校准帧必须来自训练 split，不能从评估 episode 抽取。

```bash
python scripts/validate_calib_jsonl.py data/calib-512.jsonl --require 512
```

每次采样保存随机种子、轨迹 ID、帧索引、任务套件和指令。避免只保存图片路径而丢失抽样证据。

### 7.2 生成 proxy

`sensitivity_hessian_proxy.py` 使用 `local_files_only=True`，因此先把对应 checkpoint 下载为本地目录，再传目录路径。

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir /root/autodl-tmp/qvla-repro/models/openvla-oft-spatial
test -f /root/autodl-tmp/qvla-repro/models/openvla-oft-spatial/config.json
test -f /root/autodl-tmp/qvla-repro/models/openvla-oft-spatial/dataset_statistics.json
```

如果 Hugging Face CLI 的新版命令为 `hf download`，只替换命令名，`--local-dir` 和模型 ID 保持不变。下载后记录目录大小和文件清单，防止 LFS 文件未完整获取。

```bash
cd /root/autodl-tmp/qvla-repro/src/QVLA/openvla-oft
python qvla/sensitivity_hessian_proxy.py \
  --pretrained_checkpoint /root/autodl-tmp/qvla-repro/models/openvla-oft-spatial \
  --calib_jsonl /root/autodl-tmp/qvla-repro/data/calib-512.jsonl \
  --out_path /root/autodl-tmp/qvla-repro/outputs/proxy-spatial.pt \
  --bits 0,2,4,8,16
```

验收：输出中每个目标层包含 `proxy_0/2/4/8/16`；通道数等于层的输出维度；数值有限且非负；projector、action head、LM head 不在目标层集合内。

### 7.3 分配平均位宽

分别产生 W8 和 W4 门控：

```bash
python qvla/assign_gates_from_sensitivity.py \
  --proxy_pt /root/autodl-tmp/qvla-repro/outputs/proxy-spatial.pt \
  --bits 0,2,4,8,16 --target_avg_bits 8 \
  --out_json /root/autodl-tmp/qvla-repro/outputs/gates-w8-spatial.json

python qvla/assign_gates_from_sensitivity.py \
  --proxy_pt /root/autodl-tmp/qvla-repro/outputs/proxy-spatial.pt \
  --bits 0,2,4,8,16 --target_avg_bits 4 \
  --out_json /root/autodl-tmp/qvla-repro/outputs/gates-w4-spatial.json
```

验收：`final_avg_bits <= target_avg_bits` 且接近目标；所有 gate 只取候选集合；通道总数一致；保存 bit histogram。官方 README 示例遗漏了某些候选 bit，严谨实验使用论文完整集合。

### 7.4 Fake-quant rollout

实际仓库文件名是 `qvla/run_eval.py`。先用 1 trial：

```bash
python qvla/run_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/qvla-repro/models/openvla-oft-spatial \
  --gates_path /root/autodl-tmp/qvla-repro/outputs/gates-w4-spatial.json \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --seed 7 \
  --local_log_dir /root/autodl-tmp/qvla-repro/logs/fake-w4-spatial-smoke
```

再按 10、50 trials 扩大。BF16、W8、W4 必须使用相同初始状态和 seed。记录的是“fake-W4 精度”，不能记录为“INT4 推理”。

## 8. R3：独立重建论文动作敏感度

论文方法需要比较完整精度动作 `A*` 与量化单通道后的动作，并用动作空间误差排序。建议实现顺序：

1. 固定 32 个样本，只选择一个 Linear 层的 16 个输出通道。
2. 对每个通道分别应用 16、8、4、2、0 bit 扰动。
3. 计算 teacher-forced action MSE，验证低 bit 通常产生更大误差。
4. 用 autograd 计算动作对该通道输出的 Jacobian-vector product，构建一阶 proxy。
5. 检查 proxy 排名与真实逐通道 MSE 的 Spearman 相关性。
6. 扩展到全部目标层和 512 条校准轨迹。
7. 再运行论文的相邻降级贪心算法。

通过标准：实现确实消费动作标签或 BF16 teacher action；敏感度定义和论文公式一致；使用 asymmetric per-row quantization 时明确保存 scale/zero-point；与官方 Hessian proxy 分开命名和比较。

## 9. R4：真实 INT4 与性能验证

### 9.1 先限定可实现格式

任意通道混合 0/2/4/8/16 bit 很难由一个高效 GEMM 直接执行。第一版部署建议按 bit 对输出行稳定分组：

- 0-bit 行直接移除；
- 2/4/8-bit 行分别打包并调用独立 kernel；
- 16-bit 行走 BF16 GEMM；
- 最后按原行号 scatter 回输出。

这比 fake quant 更接近论文设想，但需要把元数据、scale、zero-point、分组和 scatter 开销都算入显存与延迟。

### 9.2 基准规范

- 相同 RTX 4090、功耗上限和时钟状态；
- BF16 与 INT4 使用相同 batch、序列长度、图像数、action chunk、FlashAttention 设置；
- 预热至少 20 次，测量至少 100 次；
- 每次计时前后 `torch.cuda.synchronize()`；
- 同时报告 kernel latency、model forward latency、完整 episode wall time；
- 报告 packed weights、scale/zero-point、临时 workspace 和峰值显存；
- 至少重复 5 轮，报告中位数和 P10/P90。

端到端命令可先用辅助脚本采样：

```bash
python scripts/measure_command.py --output outputs/bf16-spatial-timing.json -- \
  python experiments/robot/libero/run_libero_eval.py [其余参数]
```

这个脚本记录端到端耗时和 `nvidia-smi` 峰值，只能辅助审计；精确 kernel 延迟仍需 CUDA event 或 PyTorch profiler。

## 10. 实验矩阵和停止条件

| 顺序 | 模型 | 实现 | Trials/任务 | 目的 |
|---|---|---|---:|---|
| 1 | Spatial BF16 | 官方 | 1 | 环境冒烟 |
| 2 | Spatial BF16 | 官方 | 10 | 估算成本和检查基线 |
| 3 | Spatial fake-W8 | 发布 QVLA | 1/10 | 检查量化链路 |
| 4 | Spatial fake-W4 | 发布 QVLA | 1/10 | 检查门控和精度 |
| 5 | 四套 BF16 | 官方 | 50 | 论文基线 |
| 6 | 四套 fake-W8/W4 | 发布 QVLA | 50 | 发布实现精度 |
| 7 | 四套 action-W8/W4 | 自建论文法 | 50 | 论文算法验证 |
| 8 | W4A4 packed | 自建内核 | benchmark | 性能验证 |

出现以下任一情况立即停止完整评估：BF16 pilot 偏差超过 5 个百分点；日志有 episode 异常；模型/评估 checkpoint 不匹配；峰值显存持续增长；量化 gate 通道数与模型不一致。

## 11. 最终复现报告应包含

- 论文、QVLA、LIBERO、checkpoint 的不可变版本标识；
- AutoDL 主机、GPU UUID、驱动、CUDA、Python 和完整依赖；
- 校准样本清单与随机种子；
- 每个 suite 的成功数、episode 数、成功率和置信区间；
- BF16 与量化模型使用的相同初始状态证明；
- fake quant、packed quant 和理论内存估计的明确标签；
- 延迟预热、同步、重复次数和统计量；
- 与论文目标值的绝对差和相对保持率；
- 当前不能复现的主张及其原因。

完成上述记录后，即使某些论文数字因代码未公开而无法达到，最终结论仍然是严谨、可审计且可用于后续 INT4 Triton 项目的。
