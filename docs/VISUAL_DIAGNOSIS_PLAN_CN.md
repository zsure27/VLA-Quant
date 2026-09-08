# AWQ W2A16 / SQ W4A4：高效可视化与修复决策

先执行 [基线审计与门槛](BASELINE_AUDIT_20260908_CN.md)。以下是在一致的输入、模型、后端和随机协议下解释量化退化的方法，不把任何假设提前当作事实。

## 1. 分三层投入算力

| 阶段 | 最小实验 | 主要回答 | 何时停止 |
|---|---|---|---|
| G0 基线门槛 | 官方源码/依赖、轨迹划分、BF16 重复、仅平滑 | 实现/数据是否已经改变基线 | 任一检查失败就停，先回传日志 |
| G1 离线定位 | 8 帧留出，W/A、视觉/语言、projector oracle | 误差最先出现在哪里，哪个接口值得修 | 不同时扫 alpha/group/层保留/训练超参 |
| G2 因果细化 | 少量 attention/MLP/patch 消融、单 block 恢复；再扩至 32–64 帧 | 哪个扰动的移除真正恢复动作 | 多个小样本排序冲突则增加覆盖，不凭一张图决定 |
| G3 修复验证 | 校准轨迹训练/调参，独立留出，20 配对回合→500 回合 | 改善是否跨状态/闭环成立 | BF16 环境异常、输出不有限、动作失控就停 |

每次只加载一个 7B 模型；BF16 教师缓存到 CPU/磁盘，各候选新进程复用。先看动作读出位置与每层摘要，不保存全模型全 token 全 head 的 attention。所有 bit、group、训练步数与校准帧数都写进 manifest。

## 2. 第一轮矩阵及图的读法

### SQ：先拆权重误差与激活误差

教师 BF16 → BF16 重复 → 仅平滑 W16A16 → W4A16 / W16A4 / W4A8 / W4A4 → vision-only / language-only → projector oracle / vision-only oracle。

其中仅平滑与 W/A 消融共用同一份 SQ absmax、alpha 和完整平滑分组。不要让“少量化了一个分支”同时变成“用了不同的平滑规则”。

- W4A16 接近 BF16，而 W16A4 已崩：优先检查激活异常值、量化步长、归零和模态差异。
- 仅平滑就产生明显动作误差：先查 Norm→Linear 共享输入、scale 维度、BF16 数值误差、模型类/后端；不应先训练掩盖问题。
- 视觉单独差、语言单独正常：优先视觉校准覆盖、patch 粒度、视觉特征进入 projector 前的误差。
- 语言单独差：继续分 attention 与 MLP，不能把问题都交给 projector。

### AWQ：W4 正对照与 W2 独立校准

每个位宽分别跑 full / vision-only / language-only；W2 另加 full + projector oracle。W4 与 W2 必须分别搜索。视觉仍为显式适配，报告里不要把系统整体写成官方原生 AWQ-VLA。

- 新 W4 也明显异常：优先查官方 block 缩放、clip 重放、输入上下文、group 与 checkpoint；暂不解释 W2 极限。
- W4 正常、W2 失败：先比较动作端敏感层，而不是只看权重 MSE。
- LLM 局部误差小但动作误差大：提示非线性/残差传播或动作敏感方向放大，下一轮用单 block 恢复找因果证据。

### 自动生成的图

| 输出 | 横/纵轴与分组 | 正确解释 |
|---|---|---|
| `*-llm-depth.png` | 深度 × 相对特征 MSE，主图/腕图/文字/动作读出分面 | 找误差首次跃升和后续放大；阴影是样本范围，不是置信区间 |
| `*-vision-depth.png` | 两视觉编码器 × 两相机的深度误差 | 区分编码器与视角覆盖问题 |
| `action-coordinate-rmse.png` | 用例 × 归一化 7 维动作误差 | 避免把平移米数与旋转量直接混合成一条“总误差” |
| `*-activation-zeroing.png` | 非零输入被舍入为零最多的 20 个目标 | 是局部异常值线索，不是动作因果敏感度排名 |
| `*-attention.png`（可选） | 深度→真实概率 JS，层×模态注意力质量变化 | 查看 action queries 是否丢失主/腕图像或文本信息 |
| `summary.csv` | 每个用例的动作 MSE、实际夹爪开合分歧 | 用于配对比较，不是成功率 |

连续 hidden features 不是真实 token 概率，不能随意对 channel 做 softmax 后把 KL 当作语言分布变化。只有 attention 探针里通过真实 Q/K/掩码得到的概率才报告 JS/entropy。夹爪分歧采用环境执行前的 `sign(2*x-1)`；连续夹爪数值不同不等于开合决策不同。

## 3. 第二轮：用最少实验区分 attention、MLP、patch

保留第一轮 SQ 全部 W4，分别让 A4 只作用于：

1. `attention`：qkv/q/k/v/o/视觉 attn.proj。
2. `mlp`：gate/up/down/视觉 fc1/fc2。
3. `smoothed` 与 `unsmoothed`：检查 Norm 平滑覆盖之外的位置。
4. `no-patch`：仅将 patch Conv 激活保留 A16。
5. `--activation-math-fp32`：同位宽、同粒度，仅把激活量化计算提升 FP32，排查数值计算问题。

例：在仓库根目录、激活环境后，以第一轮 SQ profile 和 teacher 为参照：

```bash
python diagnostics/probe.py --mode smoothquant \
  --checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial" \
  --samples-dir "$ROOT/calib/action-space-balanced/libero-512" \
  --official-root "$ROOT/src/official-quantization" --targets-file configs/qvla-connected-422.txt \
  --profile-dir "$OUT/profiles/sq" --teacher-dir "$OUT/teacher" \
  --num-samples 8 --offset 64 --activation-scope unsmoothed --output "$OUT/sq-a4-unsmoothed"
```

如果教师使用了 `ATTENTION_LAYERS`，上例也必须传同样的 `--attention-layers`。其他路径、seed、N/OFFSET 均保持一致。

这些集合层数、参数量和激活规模不同，不能直接以总 MSE 判定某类算子“天生更敏感”。细化时从前述图选 4–8 个候选 block，做两个方向：

- **一次扰动**：BF16 只量化一个 block，观察动作变化。
- **一次恢复**：全量化仅恢复同一个 block，观察动作恢复。

定义离线动作恢复率 `R = 1 - E_restore / E_full`，`E` 为同一批留出样本的归一化动作 MSE。`E_full` 接近 0 时不定义比率，负恢复率如实保留。必须说明恢复了多少权重、A16 token/层与额外位宽；混合精度改善不能继续叫均匀 W2/W4A4。

当前脚本直接支持的是范围级消融；**任意单 block 恢复尚未封装**。尤其 AWQ 的 V→O、up→down 跨 Linear 缩放不能只换回一个原始矩阵：恢复参数必须处于同一个缩放坐标系，否则“恢复”本身就会破坏函数。看到第一轮结果后再对选中的完整 block 实现/测试，避免一次写大量未验证模型手术。

## 4. projector 是否值得微调：必须有因果对照

对同一个量化候选、同一图像/指令/proprio，把 projector 的输出替换为 BF16 教师输出。保持其余量化部分不变。

| 现象 | 下一步 |
|---|---|
| vision-only + oracle 能恢复 | 说明替换接口链路正确；视觉→LLM 接口可作为修复位置 |
| full + oracle 也明显恢复 | 优先尝试小规模 projector 特征/动作蒸馏 |
| vision-only oracle 有效，full oracle 无效 | 语言量化是独立瓶颈，projector 单独修复不够 |
| 特征 MSE 大，但动作正常 | 存在动作不敏感的误差方向，不必追求所有通道逐点一致 |
| projector MSE 降低，但动作/rollout 更差 | 特征重构损失与控制目标不一致，增加动作监督并检查验证集 |

oracle 不是可部署模型，也不是“微调一定可达到的上界”：它提供了量化视觉中可能已经丢失的信息。

有证据后才尝试：冻结视觉/LLM，只训练 projector 或小残差适配器。基础损失可写为 `L = λf·相对特征MSE + λc·(1-cosine) + λa·归一化动作L1`。使用原始 BF16 模型作为教师；校准训练轨迹用于拟合，诊断留出用于选方案，最终正式评估状态不得用于训练。

冻结 LLM 参数不代表把 LLM 前向放进 `no_grad()`：动作蒸馏需要梯度通过 LLM 回到 projector。SQ A4 路径还必须有经过验证的 STE；本仓库目前的 PTQ hook 会主动拒绝这种未实现的训练路径。不要通过删掉保护来“让训练跑起来”。

## 5. 证据到修复策略的对应表

| 实测证据 | 优先尝试一个改动 | 必须保留的对照 |
|---|---|---|
| 两相机统计/误差强烈不均衡 | 均衡相机、任务与动作阶段校准，增加困难状态覆盖 | 相同帧预算、独立留出 |
| A4 高 max/RMS、非零大量归零 | 校准 alpha 小网格或保护已定位输入为 A8/A16 | W4A16、原 alpha、实际平均位宽 |
| MLP/down_proj 最敏感 | 完整 block 局部蒸馏或少量 MLP LoRA | attention-only 同等训练预算 |
| action→视觉 attention 质量下降且 oracle 恢复 | projector 蒸馏，必要时加关系/attention 蒸馏 | 特征损失单独、动作损失单独、未训练量化 |
| attention 概率 JS 大且恢复该 block 动作改善 | 选中的 attention block/LoRA 微调 | 正确 mask/RoPE；不是仅匹配 Q/K 原始投影 |
| AWQ W2 独有强退化 | group128→64→32 或 clip on/off（分次） | 同校准数据、同 held-out，报告元数据成本 |
| 离线动作很好但闭环差 | 训练演示里的状态阶段/困难状态覆盖，短时序动作监督 | 不把单步 MSE 当成功率代理的充分条件 |

alpha、层保护和 projector 不要同时改。一次只改变一个因素，最多从 2–3 个候选里选一个进入更大样本/rollout。先解决错误实现，再优化低比特本身的困难。

## 6. 建议回传的最小结果包

第一批请只跑 `STAGE=controls`，回传 runtime/split 检查、self-test.log、repeat 与 smooth-only 的 `metrics.json`、manifest 以及 figures。确认后再跑两条主线。

后续回传全部用例的 `scope.json`、`metrics.json`、manifest、图和校准 summary/log；attention 开启时带相应图。教师 `.pt`、模型权重和大型 profile 留在服务器，不需要上传聊天或普通 GitHub git 对象。

最终每次实验记录：问题假设 → 唯一改动 → 控制门槛 → 留出动作/特征指标 → 同状态配对 rollout → 是否晋级。所有未跑的格子写“待测”，不填估计成功率。
