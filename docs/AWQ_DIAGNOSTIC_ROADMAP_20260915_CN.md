# AWQ W2 诊断与 PEFT 恢复：结合本轮实测的阶段计划

## 当前证据

014，LIBERO Spatial 任务 0–9，各一个官方初始状态，paired seed 0。所有七次评测的逐任务种子与初始状态 SHA-256 一致。原 BF16 9/10；同量化入口但完全不应用 scale/clip/量化的 BF16 控制仍为 9/10；W4 原始与动作记录复测均为 8/10，额外失败都是任务 9。重复运行用于检查一致性，不是新增独立样本。

| 配置 | 语言 | 主视觉 / 融合视觉 | 成功 |
| --- | --- | --- | ---: |
| BF16 同入口 | BF16 | BF16 / BF16 | 9/10 |
| AWQ W4 | W4/G128 | W4/G128 / W4/G128 | 8/10 |
| 当前全 W2 | W2，去语言 clip | W2/G64 / W2/G128 | 0/10 |
| W2 语言隔离 | W2，去语言 clip | BF16 / BF16 | 1/10 |
| W2 视觉隔离 | BF16，未应用语言 block scales | W2/G64 / W2/G128 | 10/10 |

这组证据将优先级指向语言侧。它不证明视觉完全无损，也不证明视觉 W2 优于 BF16。语言隔离仅任务 8 成功；BF16 与 W4 均在任务 4 失败。当前排除的 action head、projector、proprio projector、norm 和 embedding 不新增量化；AWQ 算法本身的等价 block scale 变换仍按其坐标配对应用，不能误称这些参数完全没有数值变换。

十个共享初始观测上的首个动作块，相对同入口 BF16 的平均 L1：W4 **0.0117178**，语言 W2 **0.1268062**，视觉 W2 **0.0675469**。这是反归一化后的原始策略动作块、夹爪阈值/翻转之前的 teacher 对照；不是 GT action L1，不是不同闭环轨迹逐时刻相减的误差。视觉 W2 的较大初始动作偏差仍能完成任务，再次说明单一 MSE/L1 不能代替闭环。

成果目录：`results/awq-visual-diagnostics-20260915/`，含 CSV、JSON、成功矩阵、真实回放帧、动作与本体状态曲线。最初诊断记录在 `get_action` 修改输入后保存，因此其中 state 是归一化后的本体状态；图表已明确标注 normalized，不能解读成米制物理轨迹。后续代码在调用前复制原始 state，并写入 `state_space`。原始日志与运行时 evaluator patch/hash 保留，不覆写历史数据。

## 接纳用户方案后的顺序与门槛

**Phase 0 尚未完成，不进入恢复训练。** 当前只完成 BF16/W4/W2 的十状态模拟量化筛选及复测。W4 下一步先扩大预先固定的独立初始状态（例如每任务 10 个，共 100），与 BF16 配对报告差值和置信区间，单独核查任务 9 的抓取/释放阶段。还需记录同步 CUDA 的预热后延迟、峰值显存、同观测 teacher action L1、动作 token hidden cosine。现有 CSV 的未测值留空。W3 需要独立校准及位宽契约扩展，不能把 W2/W4 profile 改标签；当前没有经过验证的 packed 实现，fake/real parity、真实模型大小与加速不作已完成报告。

**Phase 1 已有粗粒度探索证据。** 基线门槛通过后，先拆语言 q/k/v、o、gate/up、down 的作用，再拆 DINO/SigLIP。继续使用相同初始状态；先定位，不同时调整多个配置。当前验证测试覆盖 224/198 目标分区、视觉隔离不施加语言 scales、BF16 控制不改权重。

**Phase 2 的回退必须保持 AWQ 坐标一致。** 在完整 W2 上做 block/group rescue 时，不能直接把原始 BF16 矩阵塞入已重缩放的 block；应在同一 AWQ 变换坐标下恢复权重，或从原始模型重新构建完整对照。先用离线同观测特征/动作误差筛选少量候选，再做闭环 rescue。模拟量化各层仍由 BF16 承载，实际显存增量可能接近 0；此时不能用它作 rescue_score 分母。报告 Δsuccess 和理论打包存储增量，等真实 packed 实现后再给实际 MB 分母。

**Phase 3 保留先前策略，但如实命名。** 语言无剪裁、视觉 G128→G64 是 PTQ 配置搜索，不是 PEFT。优先在敏感语言模块搜索 G128/64/32、clip 与量化零点策略；每种方案保留新的 profile、校准样本清单与源代码哈希。保持 VLA 图像/指令/proprio 校准，并区分全局和任务均衡采样。搜索集与最后评估集隔离。

**Phase 4 有 adapter，基座仍待核实。** 服务器 checkpoint 下 `lora_adapter/adapter_model.safetensors` 与 `adapter_config.json` 存在，rank=32、alpha=16，配置中的 `base_model_name_or_path` 为 null，模型目录目前仅见合并 OFT checkpoint。需取得有来源指纹的原始基座，验证与 adapter 合并后能重现当前权重，再比较 Q2(Wbase+Δ) 与 Q2(Wbase)+ΔBF16。不能把 Wmerged−Δ 当作未经说明的原始基座，BF16 合并舍入也必须进入误差说明。delta 范数、bin 宽、有效码变化率及动作效果一起报告。

**Phase 5–7 以少量参数修复为主。** 敏感模块的 W4/BF16 回退作为诊断与效果上界，平均位宽按真实参数数目加权并包含 scale/zero-point 开销。Scale-PEFT 冻结整数码 q 与明确的 zero-point，训练选定权重量化组的 log dequant-scale 增量；AWQ 通道重缩放向量与量化反量化步长不是同一个参数，不能混用。用原 BF16 teacher，action GT/teacher L1、动作 token hidden cosine 与尺度正则；随后只在敏感模块测试 rank 4/8/16 的 BF16 recovery LoRA。保持原 action/projector 排除方案，报告训练参数量及增量存储，不将骨干全参更新称为 PEFT。

**Phase 8–9 是有条件的后续。** 静态 Scale-PEFT 确有闭环收益后，才训练上下文专家并做 cross-context 矩阵。当前 checkpoint 仅 Spatial，直接拿 Object/Goal/LIBERO-10 的损失比较可能混入任务能力差异；先在具有相应教师和基线支持的上下文中证明专家互补。如果没有互补就停止，不实现路由。第一版只训练小型 Top-1 scale expert router；不做 Top-2、动态 bit、完整 scale hypernetwork 或在线写回。比较 global scale、oracle scale、learned router，并明确 oracle 的已知任务标签不能冒充部署可得输入。

最终九个研究问题中，目前只有“语言侧是优先瓶颈”获得本轮小样本支持。最敏感层、LoRA delta 是否被吞掉、最小平均位宽、Scale-PEFT/LoRA 增益与 Router 必要性均仍待实验，不能提前下结论。
