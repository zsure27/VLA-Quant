# P2.5：Recovery-LoRA 闭环对齐诊断与重对齐计划

更新：2026-09-26。本文件覆盖此前把 data80 解释为单独“扩大训练覆盖”，以及把当前
50 回合闭环分片解释为 W4 恢复率测试的做法。当前研究问题固定为：

> 为什么 PEFT 能在固定观测上显著接近 BF16，却没有同步改善闭环控制；Recovery-LoRA
> 能否在自身动作诱导的状态分布上继续保持正确？

在回答这个问题前，不启动 Contextual Routing、P4 专家、静态 W4 block 搜索、rank 网格或
更多 SVD 变体。

## 1. 固定边界

- backbone：`awq-w2a16-12l-mixed-spatial-v1`；
- 唯一目标：语言 blocks 18–19 的 14 个 Linear；
- Recovery-LoRA：rank8，Response-SVD 只作为初始化；
- 视觉、其他语言层、clip、保护模块和评测器契约保持不变；
- `offline_final_holdout` 保持封存；
- Scale-PEFT 保留为负基线，不再搜索超参数，也不作为首个专家族。

历史 12L=412/500、14L=431/500 与最新完整复跑 411/500、430/500 存在各一次成功的偏移。
在来源、模型、量化 profile、评测器、初态清单和 seed 审计完成前，报告使用“约 411/500”与
“约 430/500”，不得混用精确值。审计完成后只保留一个 canonical 口径，旧值明确标为不同
session 的历史证据。

## 2. data80 的正确解释

| 单元 | 唯一训练轨迹/帧 | 优化步数 | 每个选中帧的有效曝光 | 状态 |
|---|---:|---:|---:|---|
| A | 16 | 200 | 12.5 | 已有 |
| B | 16 | 1000 | 62.5 | 待跑 |
| C | 80 | 200 | 2.5 | 待跑 |
| D | 80 | 1000 | 12.5 | 已有 data80 |

D 相对 A 同时增加了轨迹覆盖和总优化量，因此当前唯一成立的结论是：**更大的轨迹覆盖加上
更大的总优化曝光带来了更强的固定观测泛化**。D 的 normalized action MSE 为 0.02284，
31/32 历史帧改善，gripper disagreement 为 0.05078；这些结果不能单独归因于覆盖率。

B/C 继续固定 rank8、Response-SVD、Smooth-L1(beta=0.1)、学习率、优化器、blocks18–19
和 batch 语义。四个单元先进入轨迹级 `router_dev`，不立即全部进入闭环。

## 3. 轨迹级 router_dev 门禁

历史 32 帧降级为回归与调试集。新的开发门禁对每条 `router_dev` 轨迹按预注册位置
`p={0.1,0.3,0.5,0.7,0.9}` 抽样，索引为 `round(p*(T-1))`；重复索引去重。抽样器保存：

- trajectory ID、数据集顺序、源路径、指令、原始步数、步索引和实际归一化位置；
- 每个样本、轨迹切分和程序的 SHA256；
- 当前帧之后的 8 步动作块，末端不足时重复最后一步，仅用于字段和语义审计；
- `role=router_dev` 与 `holdout_touched=false`。

主报告同时包含 frame mean/median、每轨迹 median、改善轨迹比例、最差十分位退化、最大轨迹
退化、位置/旋转/夹爪连续误差、夹爪阈值分歧与 margin。单个大误差帧不能单独决定门禁。
Response-SVD 的所有 X 继续只来自 `peft_train`，并保存轨迹与帧哈希。

## 4. 当前 50 回合的证据等级

既有分片中 12L=43/50、14L=43/50，观察到的静态 headroom 为零。正在运行的 data80
50 回合因此只称为 **closed-loop stability pilot**，用于发现灾难性退化、很大的正收益或明显
的夹爪/控制不稳定。若差异很小，结论为不确定；当 `S14-S12=0` 时恢复率字段必须为 `null`。

后续分成两种视图：

1. 冻结的宽覆盖配对开发成功率，用于开发证据；
2. 12L 失败且 14L 成功的历史 episode 子集，只用于 blocks18–19 敏感机制诊断，明确标记
   post-hoc，不把其成功率当作无偏 benchmark。

最终结论仍需另行冻结新的闭环协议。

## 5. P2.5 同观测 on-policy 诊断

首个 P2.5 分片只运行 data80 Recovery-LoRA 学生。每次策略在学生实际观测
`o_t^student` 上重新推理时，保存：

- 主相机、腕部相机、proprio、指令与观测哈希；
- 学生原始 8×7 动作块；
- block17 输出的 token mean 与最后 token 向量；
- episode、query 序号、实际执行步起点和最终成功/失败。

学生运行结束后释放显存，单独加载冻结 BF16，在序列化的同一观测上查询教师。这个两阶段
实现避免两套 7B 模型同时驻留。每个 query 保存教师动作、H17 摘要和文件哈希，并报告：

- XYZ、rotation、gripper continuous 的误差；
- 按现有 `sign(2*x-1)` 语义计算的夹爪分歧和两侧 margin；
- 8 个实际执行时间位置的误差；
- episode 内首次夹爪分歧；
- 首次“大策略分歧”只在阈值由 `router_dev` 预注册后报告；此前保持 `null`，不事后造阈值。

已核实当前评测器把完整 8 步动作块放入队列，8 步全部执行后才重新推理，属于 open-loop
chunk，不是逐步 receding horizon。因此必须先报告逐 step 误差，再根据其与闭环失败的关系
决定是否预注册时间权重。

P2.5 比较三种分布：`peft_train` 固定观测、`router_dev` 固定观测、LoRA 自身访问观测。
若前两者低误差而第三者误差随 episode 明显增长，covariate shift 成为首要解释。

## 6. 确认 covariate shift 后的唯一下一训练项

只在 P2.5 支持 covariate shift 后，执行同预算 student-state teacher relabeling：

1. 在训练 reset 条件上运行当前学生；
2. 保存学生访问的观测；
3. 冻结 BF16 在相同观测上生成动作标签；
4. 将 `D_student={(o_t^student, a_BF16(o_t^student))}` 与原 `peft_train` 混合；
5. 保持 exact-12L、blocks18–19、rank8、参数量和初始化策略不变，重训或续训；
6. 对比 offline-only 与 student-state-distilled 在 router_dev、同观测 on-policy 分歧和配对闭环上的结果。

不增加 rank、专家数或目标层，也不使用未来成功、GT future action 或 post-expert 特征作输入。

## 7. Contextual Routing 的新门禁

shared PEFT 失败可能来自：

- H1：目标或访问分布不匹配，各上下文需要近似相同的修复；
- H2：上下文之间的修复方向冲突。

P2.5 先处理 H1，再以 task/phase 分组的梯度 cosine、Response-SVD 子空间、adapter response 和
误差向量诊断 H2。高对齐说明 Router 缺少动机；稳定的低或负 cosine 与不同主子空间只构成
专家专业化的诊断依据，不等于 Router 证据。

只有 shared Recovery-LoRA 有稳定闭环收益，或 H1 已处理而 H2 仍有强、跨轨迹可复现证据，
才允许提出 P4 两个同预算 rank8 专家。H17 Top-1 Router 保持锁定，直到固定专家矩阵、
BestFixed–Oracle gap、跨 seed 互补性和因果可预测性全部成立。

## 8. 当前连续执行顺序

1. 不打断正在运行的 data80 50 回合稳定性试跑；
2. 保存原始输出并按“稳定性”重新解释；
3. 完成 411/412 与 430/431 的来源审计；
4. 建立 `router_dev` 轨迹级数据和 exact-12L 基线；
5. 运行小规模 P2.5 同观测诊断；
6. 完成 B/C 两个缺失单元并在 `router_dev` 比较 A/B/C/D；
7. 依据 P2.5 决定是否做 student-state distillation；
8. 不满足 P4 门禁时不训练专家，不实现 Router。

每项实验必须明确回答以下至少一个问题：学生访问状态上能否恢复、shared 失败是否来自上下文
冲突、若冲突成立 H17 是否可能预测正确专家。只降低固定帧 MSE 且不回答其中任一项的工作不进入
优先队列。

