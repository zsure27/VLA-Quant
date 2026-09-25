# PEFT 与 Contextual Routing 主线执行协议

更新：2026-09-25。本文件根据已完成的 Spatial500 混合精度实验修订执行顺序，并覆盖旧计划中继续扩展静态 W4 block 搜索的建议。研究因果链固定为：

`static low-bit backbone → shared PEFT recovery → expert complementarity → observable-context prediction → contextual routing`

任何新实验必须注明它检验 P1、P2、P4 或 P5 中的哪一环。仅搜索新的 W4 block 子集不进入执行队列，除非用户明确重新开启 P3。

项目主线仍是 VLA 快速量化微调，首要量化对象是 AWQ-W2A16。Contextual Routing 的直接动机是纯 2bit 精度失效后的轻量能力恢复；SmoothQuant 只在该主线得到有效性证据后作为独立扩展。当前唯一默认 backbone 的机器可读定义是 [`awq_w2a16_12l_mixed_spatial_v1.json`](../configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json)。若后续证据支持改变 baseline 或 backbone，必须新增版本化配置、配对复核并记录替代原因，不能直接改写 v1。

## 1. 冻结 P3 静态混合精度搜索

三种固定参照共享以下配置：语言其余低位 block 为 W2 G64，attention V/O 不做额外 clip，MLP 保留冻结配方中的 clip；DINO W2 G64；SigLIP W2 G128；既有保护模块继续保持高精度；评测和 fake-quant 契约不变。

| 角色 | 语言 W4 blocks | Spatial500 | 用途 |
|---|---|---:|---|
| 16L | 8–23 | 434/500（86.8%） | 较强静态恢复参照 |
| 14L | 8–15、18–23 | 431/500（86.2%） | 静态部署膝点候选 |
| 12L | 8–15、20–23 | 412/500（82.4%） | 主 PEFT 恢复实验底座 |

禁止自动运行 13 层、11 层、其他 14 层组合或更多粗粒度 W4 岛搜索。14L 不是唯一 Pareto 最优点；它只是当前数据支持的静态部署候选。

12L 相对 14L 只将 blocks 18–19 从 W4 改为 W2，却下降 19/500；这里有清晰且足够的修复空间。核心问题固定为：**能否让 blocks 18–19 保持 W2，用远小于两个完整 W4 block 的 PEFT 存储接近 14L？** 其他量化设置在该实验中不得改变。

12L 的语言精度布局固定为 `0–7 W2 | 8–15 W4 | 16–19 W2 | 20–23 W4 | 24–31 W2`；所有 W2 语言层使用 G64。Projector、Action Head、Proprio、Norm 及既有保护模块继续保持 BF16/高精度。

## 2. P1：训练数据与参数化契约

P1 完成前不得报告固定码 Scale-PEFT 结果。

1. 盘点官方 LIBERO 训练轨迹、字段和许可；核对图像、指令、proprio、8 步动作块、坐标归一化和夹爪语义。
2. 按轨迹划分 `PEFT-train / router-dev / final-test protocol`，不得按帧随机切分。
3. Spatial 官方 states 0–49 已反复用于模型选择，只作为历史/开发配对证据，不再称盲测，也不得作为训练轨迹。
4. 在当前 group、padding、clip、`input_scale` 和 BF16 舍入语义下重建并冻结 AWQ `q/z/Δ`。
5. 验证 `δ=0` 逐层和动作输出复现当前 fake-quant 路径；保存训练前后 q/z 哈希。
6. 验证只有允许的 scale residual 获得有限非零梯度，并验证保存/重载一致。
7. 若无法复现，只能命名为“有效权重乘性残差代理”，不得称固定码 Scale-PEFT。
8. LoRA 必须验证冻结 base、可训练 A/B、标准随机 A/零 B 的零输出前向、一步有限梯度和保存/重载一致性。

最终评估协议须在模型选择结束前预注册，使用真实改变的 reset、seed 或受控扰动，并保存初始观测哈希。只有 PEFT 类型、目标层、rank、专家数和 Router 特征全部冻结后才能使用。

## 3. P2：同一 12L 底座上的 shared PEFT

首轮目标模块只允许语言 blocks 18–19，不从全部 W2 层起步。

### D1：Scale 修复

比较 `12L`、`12L + shared Scale-PEFT(blocks18–19)` 与 `14L static W4`，测量 scale-only 修复能追回多少 W4 收益。

### D2：Recovery LoRA

比较 `12L` 与 `12L + Recovery LoRA rank8(blocks18–19)`。标准零输出初始化和 Response-SVD 初始化必须使用完全相同的训练轨迹、步数、优化器、loss、seed 和目标模块。Weight-SVD 仅在资源允许时作为次要初始化。Response-SVD 只是初始化方法，不是运行时专家。

两类实验都必须报告零步 loss、训练曲线、位置/旋转动作误差、夹爪误差或 margin、action-token hidden error、闭环开发成功率、可训练参数、adapter 字节、训练显存和时间。离线 MSE 不能代替闭环判断。

若 Scale 和 LoRA 都有稳定收益，再在同一 12L 底座、同一模块和数据契约上做 `12L / +Scale / +LoRA / +Scale+LoRA` 的 2×2。

## 4. P4：专家互补性门槛

shared PEFT 未在留出闭环上稳定改善前，不训练专家或 Router。

首个专家库只有两个专家。二者必须共享 12L 底座、blocks 18–19、PEFT 类型、rank、参数量、总训练曝光、优化器和 losses；只允许训练上下文分布不同。优先预注册可解释的可观测分区，例如 approach/transport 与 grasp/place。task-ID 专家只作诊断对照。

在留出固定观测上让每个专家运行每个样本，保存 `context × expert` 动作 loss、夹爪 loss、winner 和 margin。随后让每个固定专家独立运行配对闭环，报告 shared adapter、最佳固定专家、hindsight oracle、交叉专家矩阵和跨 seed 稳定性。不得用回合未来成功作为 Router 训练标签。

若一个专家支配，或 oracle 与最佳固定专家近似相同，停止 Router 工作并保留 shared/static PEFT。

## 5. P5：可观测上下文 Router

仅在 P4 证明稳定互补后执行。由于专家作用于 blocks 18–19，首选路由特征是 block 17 后、block 18 前的因果表示 `H^(17)`：视觉 token 池化、指令 token 池化、proprio、action query 池化，以及可选的路由前量化统计。

禁止未来成功、推理时 BF16 future action、GT future action或 post-expert hidden state。首版为小型 MLP、Top-1 部署，每个 8 步 action chunk 决策一次。

必须比较最佳固定专家、shared PEFT、task-ID-only、instruction-only、完整上下文、随机 Router、打乱上下文和离线 oracle。只有完整上下文在锁定测试上超过最佳固定专家与 task-ID-only，同任务/同指令内部出现稳定有用的状态偏好，且收益超过存储、计算和切换成本，才允许提出 Contextual Routing 结论。

## 6. 下一轮执行队列

1. CPU 优先完成训练轨迹 inventory、按轨迹切分登记和泄漏审计。
2. 实现并测试 AWQ `q/z/Δ` 零残差等价；不合格则修复或改名为代理试验。
3. 验证 blocks 18–19 的 Scale 与 LoRA 冻结、梯度和重载契约。
4. 先做约 200 步 shared pilot，再做有限闭环开发评估。
5. 只有 shared 方法有稳定正收益，才进入两个等预算专家及全专家固定观测矩阵。
6. 只有专家互补跨 seed 稳定，才训练使用 `H^(17)` 的最小 Top-1 Router。

服务器运行仍遵守额度和快速收尾约定。静态 W4 搜索冻结并不删除 16L/14L/12L 结果；它们分别作为恢复上界、部署候选和 PEFT 底座长期保留。
