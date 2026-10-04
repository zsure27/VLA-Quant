# A/B 模型命名与数据使用审计（2026-10-05）

## 规范命名

名称按量化及恢复难度递增，不按成功率排序。`BF16` 保持原名；A 为不新增恢复微调的 AWQ，B 为 AWQ 加恢复微调。BF16 checkpoint 本身已有原始 OFT 微调；“不加微调”指不加本项目的恢复 adapter。所有 W2/W4 都指 AWQ 合格连接目标，projector、action head、embeddings、norms 等保护模块保持高精度；A4 也不是每个参数均为 2 bit。当前实现为高精度存储的 fake quant，不据此声称 INT2 部署压缩或加速。

混精 A1/A2/A3 的视觉统一 DINO W2/G64、SigLIP W2/G128；语言 W2/G64，W4 岛保留来源 W4 profile 的坐标、group 和 clip。语言 W2 的 attention V/O 无额外 clip，MLP 保留原 clip。

| 新命名 | 模型具体构成 | 模型测试结果 | 所用训练和测试数据 |
| --- | --- | --- | --- |
| **BF16** | 冻结 OpenVLA-OFT Spatial BF16 checkpoint，零 AWQ 目标 | 历史 **487/500（97.4%）**；本轮 **292/300（97.3%）** | 本项目无新增训练；历史官方 reset0–49；本轮同条件 reset20–49，10 任务 |
| **A0** | 全合格连接目标 AWQ W4A16，原 W4 | 历史 **486/500（97.2%）**；本轮 **294/300（98.0%）** | AWQ 校准 32 帧/32 演示轨迹；无恢复微调；与各自 BF16 cohort 配对 |
| **A1** | 原 16L：语言 W4 blocks8–23，其余语言 W2；视觉 W2 | 历史 **434/500（86.8%）**；本轮 300 未测 | 冻结 AWQ profiles，无恢复训练；历史官方 reset0–49 |
| **A2** | 原 14L：语言 W4 blocks8–15、18–23，其余 W2；视觉 W2 | **431/500（86.2%）**；另一来源 **430/500**，不合并；本轮 300 未测 | 冻结 AWQ profiles，无恢复训练；各历史 cohort 的官方 reset0–49，430/431 来源差异保留 |
| **A3** | 原 12L/C0：语言 W4 blocks8–15、20–23，其余 W2；视觉 W2 | 历史 **412/500**；后来配对 **411/500**；本轮 **244/300（81.3%）** | 冻结 AWQ profiles，无恢复训练；412/411 分属来源；本轮 reset20–49 |
| **A4** | 全合格 AWQ 目标 W2A16：语言/DINO G64、SigLIP G128；保护模块高精度 | 匹配此配方历史 **10/50（20.0%）**；本轮 300 未测。原 G128 的 0/50、双视觉 G64 的 11/50 不属于这个固定版本 | 冻结 AWQ 校准，无恢复训练；历史开发 reset5–9，10 任务；不是全项目盲测 |
| **B0** | 原 C3：A3 + blocks18–19 rank8 Recovery-LoRA，**1,249,280** 参数 | 本轮 **260/300（86.7%）**；对 A3 **25 rescue / 9 break，+5.3pp**；早期389/450；035随机流试跑85/100对81/100，9 rescue/5 break | student-state80：旧学生策略的 reset0–3，40 回合各前2查询，冻结 BF16 同观测重标；1000 步、batch1；本轮测试20–49；旧450测试5–49、035测试10–19，均分别报告 |
| **B1** | 原 LW：A3 + 全部语言 W2 blocks0–7、16–19、24–31 rank8 LoRA，**12,492,800** 参数 | 本轮 **284/300（94.7%）**；对 A3 **45 rescue / 5 break，+13.3pp**；对 B0 **30 rescue / 6 break，+8.0pp** | 与 B0 同一80观测、BF16同观测标签、1000步、batch1；测试20–49与A3/B0/BF16/A0严格配对 |
| **B2** | A3 + 冻结 B1 语言 adapter + 新视觉 rank8 增量 adapter；视觉只包含已量化的 transformer Linear | **未测；HOLD_CODE_SMOKE**。探索性计划，旧确认性视觉门禁未通过 | 计划沿用 student-state80；BF16同观测标签；训练/测试条件冻结，新增视觉范围是唯一干预；正式结果必须来自新配对运行 |
| **B3** | A4 + 新语言32-block rank8 LoRA；首版视觉 W2 冻结，不训练视觉 adapter | **未测；HOLD_CODE_SMOKE** | 计划采集 A4 在训练 reset0–3 访问的80观测并由BF16重标；不是静默复用B0/B1数据；与同条件A4/BF16/A0配对 |

上表“本轮”唯一指 `20260930-059-language-w2-all` 的 first50trace + remaining250，**300回合/配置**。原无观测 trace 的 first50-v2 与重跑 first50trace 不叠加。不能用历史500作为本轮300的分母或相减计算 headroom。A1/A2 历史尚无本轮300参照。

机器可读注册表：`configs/model_registry_v1.json`。旧日志/不可变计划的 C0/C3/LW 保留，**只对已确认版本/SHA的对应来源**解析为 A3/B0/B1，不仅凭同名文本跨会话映射，不覆盖原始证据。旧“实验阶段 A1/A2”分别写作 `EVAL_REPLICATION` / `TRAIN_ORDER_REPLICATION`，与新模型 A1/A2 区分；离线控制矩阵 A/B/C/D 也只是历史实验条件，不是新模块。Scale-PEFT 与离线 data80 LoRA 保留为历史负对照，不映射为 B0。

## 严格审计结果与修正

### 1. 已证实的历史校准污染

432条演示轨迹、52,970 transitions 的轨迹切分为298 `peft_train`、59 `router_dev`、75 `offline_final_holdout`；轨迹ID和源序号均唯一。但 AWQ 校准早于该切分。历史 W4 profile 实际使用的32个样本按 episode序号、instruction 与轨迹清单交叉核对，交集为：

- peft_train：18条；router_dev：8条；offline_final_holdout：**6条**（源序号378、1、361、186、258、44）。
- 因此75条只对新增 PEFT 训练留出，**不满足全流程未使用**；59条router_dev仍可作开发集，但不能声称与冻结PTQ校准完全无交集。
- 保留历史切分/profile/结果，不在审计中重校准或改 backbone。新增逐轨迹污染清单，后续报告必须标注暴露。其他 G64/G128 profile 必须各自再核对完整校准来源；本轮至少 W4 的交集已证实。
- 后续若需要全流程留出，必须预先排除所有冻结 profile 的校准轨迹，并另外审计 BF16/OFT checkpoint 的训练来源。剩余69条不能直接改名为最终盲测：它们的其他profile暴露和基础checkpoint训练排除尚未建立。此次只读元数据，未运行任何 holdout 前向/标签评估。

这不否定本轮同条件的 B1−A3 开发对照，但限制泛化结论。训练集切分正确不能自动证明整个训练/量化流程无泄露。

### 2. 学生重标与训练样本量

归档原件核验了80个NPZ文件 SHA 与策略可见 observation SHA，确为10任务×4训练reset×2查询，40条源回合；与295个 router_dev 原件未发现同观测交集。B1训练清单的80个文件SHA与B0相同；300测试的task/reset与训练无交集。BF16在施加学生量化之前对同一观测查询生成标签，placeholder action 不作监督。

学生状态来自旧离线 LoRA 学生，不是后来 B0/B1 自身的 on-policy 分布；不能把80观测称为80条轨迹，更不能把1000优化步当1000独立样本。batch1约12.5次重复曝光，前两查询缺乏后期抓取/纠错覆盖。80观测足以探索可恢复性，**不足以证明跨环境泛化**，对扩大10倍参数的B1尤其如此。下一轮先严格匹配数据预算做探索，再单独预注册覆盖增量与训练种子复现，不同时改多个变量。

修复：`diagnostics/probe.py` 在加载模型前要求恢复输入有role与SHA清单；离线输入必须传 `--trajectory-split` 并验证 `peft_train`，拒绝router_dev/holdout。校准、训练、probe之间使用图像/腕图/状态/指令的内容指纹比较，复制改名或修改placeholder action不能绕过检查。未来数据生成器需保持相同清单契约，否则停止，不静默退回路径检查。

这里核验的是原始观测内容和轨迹身份；真实预处理后图像/token/proprio是否相同，还必须在评测器完整调用顺序的烟雾门禁中核对。原始SHA不同不单独证明模型看到的新条件不同。

**尚待补证据：** 本机有16轨迹校准角色清单和历史80帧文件hash，但未找到继承Response-SVD **80帧**的完整源轨迹角色清单。80个student-state训练原件已通过核验，不能据此替代SVD初始化校准的来源证明。下次开机必须从持久盘恢复80帧原件/manifest，逐条对照轨迹split并通过新契约；未完成前不训练B2/B3，不用文件hash声称初始化没有留出暴露。若交集存在，保留历史结果并标记，需要新的版本化train-only校准方案，不覆盖旧权重。

### 3. backbone、adapter与版本

本地archive与评测contract校验表共同证明：B0加载SHA `67cd6a6d…335710a`，28张量、2,508,930字节；B1加载SHA `a231d7fb…8389bf8`，280张量、25,086,682字节，来自1000步e2e训练，不是10步烟雾或SVD初始化。两者rank8、7种Linear家族×目标block×两矩阵覆盖完整；仅对语言W2层附加残差。现有评测器还会核对shape与finite，禁止保护模块或W4层被错误附加。

新增 `--model-id` 的规范路径按注册表核验profile SHA集合、adapter SHA、rank、完整目标层、422量化目标及其bits/group。B0/B1还必须提供 `--recovery-training-manifest`，内容SHA锁定并排除训练reset。以后规范A/B结果必须使用此路径；没有model-id的旧命令仅标为历史/未注册诊断，不能凭文件名升级为规范结果。

B2视觉装载/冻结训练路径、B3新32-block路径尚未实现并通过烟雾，规范入口直接拒绝这两个名称。不能放宽旧20-block保护门禁来假装支持B3。下一次开机先实现版本化分支、零残差/重载/形状/梯度/10步烟雾，再配对微测与首片硬门禁。

### 4. 多模型比较与不确定性

300逐回合共同key为task、reset、初态SHA、env/model seed、protocol。旧聚合对齐通过；本次再次复核训练排除、adapter绑定并重算。收益、rescue/break、任务净差、McNemar均从共同逐回合布尔标签产生，不能独立排序或拼不同reset cohort。

- B0−A3：+16/300；任务聚类95%区间 **−1.0～13.0pp**。
- B1−A3：+40/300；任务聚类95%区间 **0.3～32.7pp**。
- B1−B0：+24/300；任务聚类95%区间 **−0.3～20.7pp**，不满足旧“下界>0”的确认性视觉门禁；B2只能按新卡探索。
- B1与BF16差−8/300、与A0差−10/300，分别−2.7pp、−3.3pp；本开发cohort已接近参照，不能再沿用旧B0的“相差超过10pp”描述B1。

修复聚合遗漏的B1对B0及BF16/W4配对翻转；公共统计函数不再把任务数10、分母300写死，拒绝重复task/reset与非布尔标签。任务重抽样区间和逐episode McNemar回答不同问题，不用一个显著性替代另一门禁。first50有完整观测轨迹，remaining250以manifest和动作trace核对；未声称全部300观测轨迹已逐帧验证。历史411/412、430/431继续分来源保留，不挑较好值当唯一canonical结果。

七项比较为补全描述性审计，不是七次新的预注册假设检验；未做多重比较校正，不据新增p值改变历史gate。

### 5. routing输入的潜在因果错误

B1在blocks0–7、16–17已有adapter，因此B1的H17位于部分adapter之后，只能作P2.5诊断，**不是选择B1之前可取得的base-only上下文**。未来Router若路由包含这些层的adapter，需单独预注册无专家的base-only上下文/更早共享特征，并计入额外前向成本；禁止用post-expert H17、未来成功或task ID作为捷径。

rescue和break同时存在说明成败翻转共存，但不是context conflict已经被证明。关键是同任务内可观测状态能否稳定预测哪个adapter更好；净差越大本身不意味着冲突越强。先验证静态B2/B3及同条件泛化，再做专家互补与context预测，Router仍锁定。

## 验证与复现

运行 `scripts/audit_model_registry.py`：仅使用本机忽略归档和Git清单，不连接服务器、不训练、不查询holdout。输出 `results/experiments/p2-shared-peft/20261005-model-registry-audit/audit.json`，包括原件SHA、污染ID、adapter元数据与7项配对比较。运行 `python -m unittest discover -s tests -p test_registry_data_contract.py -v` 验证复制观测、伪装角色、错adapter/秩、重复cohort、planned模型拒绝和安全元数据读取。当前本机没有torch，未进行新的GPU/torch重载；不能将CPU审计记成B2/B3运行烟雾通过。

配套卡：[B2视觉增量](../../../../docs/pre-run-cards/VISUAL_INCREMENTAL_LORA_20261005_CN.md)、[B3全W2恢复](../../../../docs/pre-run-cards/PURE_W2_RECOVERY_LORA_20261005_CN.md)。历史结果原件不覆盖；本次审计报告与JSON以同名reports/results目录交叉引用。
