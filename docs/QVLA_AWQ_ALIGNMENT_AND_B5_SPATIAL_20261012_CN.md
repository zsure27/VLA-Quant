# QVLA AWQ 基线核对与 Spatial 扩大评测（2026-10-12）

## 1. 核对结论

**本次完成的是本机源码、历史执行记录和指标重算；没有新的GPU结果。**

用户截图中92.5%是四suite均值。与本项目Spatial对应的是论文AWQ-W4A16 **93.0%**、FP **97.6%**，差4.6个百分点。[QVLA表2](https://arxiv.org/html/2602.03782v1#S4.T2)

| 来源与范围 | BF16 | AWQ-W4 |
|---|---:|---:|
| QVLA论文Spatial | 97.6% | 93.0% |
| 本机历史官方reset0–49，500回合 | 487/500，97.4% | 486/500，97.2% |
| 同一历史数据筛除reset0–4，450回合 | 440/450，97.78% | 439/450，97.56% |

旧A0已有完整500；差距不能简单解释为只测了300或样本太少。扩大到5–49有助于排除学生训练初态，但不会自动使AWQ成绩变为论文93%。

**验收目标是实现与协议正确、差异可追溯。不能调差模型或选择校准集来追求某个论文成功率。**

## 2. 已核实的本机AWQ

- MIT llm-awq固定提交：d6e797a42b9ef7778de8ee2352116e0f48a78d61。
- A0为W4A16、非对称zero-point量化、G128；422目标中224语言Linear、196视觉transformer Linear、2 patch Conv。
- projector、proprio projector、action head、embeddings、norm等保留高精度。
- 语言直接调用官方block scale搜索及clip primitive；视觉是声明过的自写逐Linear/Conv适配，不能称作者完整ViT AWQ配方。
- A0使用32个观测，每条来源轨迹一帧；十任务覆盖不均匀，每任务2–7帧。
- 历史实际console记录全部422目标用4bit、group128、没有撤销语言clip；未发现把BF16误标W4或漏做整套量化的证据。
- 四个核心适配源码hash与冻结profile metadata一致；原W4 profile SHA为ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4。
- 当前是pseudoquant数值前向，没有真实INT4/W2 packed部署证明。

本次新增可重跑审计脚本：
[scripts/audit_qvla_awq_alignment_20261012.py](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/scripts/audit_qvla_awq_alignment_20261012.py)。

本机执行通过：PASS_ARCHIVED_A0_PROVENANCE_AND_METRICS。它核对源码来源、声明scope、完整500唯一配对和程序异常，并重算450子集；不替代当前模型张量、GPU数值等价或论文复现。
[CPU回执](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/results/experiments/p0-foundation-baselines/20261012-qvla-awq-alignment/cpu-audit.json)

## 3. 目前无法等同QVLA论文的字段

| 字段 | 我们 | 作者公开证据 |
|---|---|---|
| AWQ实现 | 官方Llama primitive＋自写视觉适配 | 官方公开树没有表2 AWQ完整实现 |
| AWQ校准 | 32观测，来源manifest固定 | 表2 AWQ轨迹/帧数、采样和完整输入未公布 |
| 512轨迹 | 不能用来描述我们的32观测 | Appendix F明确用于QVLA敏感度/位分配；不能直接归于AWQ |
| group/clip/zero-point | G128，官方语言规则及视觉适配规则 | 作者AWQ具体字段尚未锁定 |
| checkpoint | moojink Spatial checkpoint内容hash固定 | 表2具体HF revision/file SHA未公开确认 |
| 评测随机流 | patched paired逐episode播种 | 公开OFT入口整体seed7、环境0；作者表2实际命令/seed集合未知 |
| 评测初态 | 官方0–49，有分片offset | 公开默认0–49；5–49不是论文完整原样基准 |

作者公开代码是QVLA fake quant与proxy，不是AWQ。其对称逐行量化不能冒充AWQ实现。
[官方代码](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/qvla/inject_fake_w.py)
[官方OFT评测器](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla-oft/experiments/robot/libero/run_libero_eval.py)
[作者仓库AWQ配置问题](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/8)

公开信息不足以保证精确重现93.0%。本机内部正确性和论文配方相同是两项不同结论。旧A0应保留为本项目AWQ参考，外部复现状态改为“论文AWQ配方尚未完全对齐”。

## 4. 下一次开机先做的基线桥接

CURRENT_PRIMARY_STAGE = P0 baseline alignment CONTRACT/DIAGNOSTIC。
B5方向已获用户2026-10-12授权筹备；运行仍需下面的依赖核验。旧18–19专用H17方案不直接用于全层B4。

1. 同步原生收尾helper、检查资源，锁定checkpoint、OFT组件、norm stats、LIBERO/MuJoCo/runtime与evaluator SHA。
2. 重跑现有官方primitive CPU数值契约及实际W4加载覆盖：零量化/scale-only等价、确有整数码/反量化变化、422目标位宽、clip、保护模块、profile/hash和动作head一致。
3. 同一旧A0 profile先做两条版本化协议诊断：upstream默认随机流与历史paired随机流；每条都同跑BF16/A0。保持图像、proprio、动作处理等固定，只改随机流。
4. upstream桥接从reset0顺序启动，保留默认seed7/env0；当前offset分片入口只允许paired，不能简单更改字符串冒充upstream。先微测，再最多50回合硬审计；需要完整paper协议时从0–49顺序评测。
5. 根据协议差异判断是否需要校准/视觉配方的单变量对照。作者AWQ原配方未公开时，采用公开MIT配方须明确标注复现假设；不能把假设写成作者配置。
6. 只有实际模型、输入、量化张量与评测语义通过，才固定新版A0和后续四配置协议。分数方向不决定门禁；残余论文差异如实记录。

若修改的只是A0 W4参照，可新增A0版本，保留旧B4。若改变W2底座、校准、保护范围、checkpoint或动作语义，则旧B3/B4不能自动认作适配；需新版本训练/验证，旧artifact不可覆盖。

## 5. reset0–4与5–49：合理用途和限制

- 现有student-state80实际来自reset0–3，未用reset4。排除0–4是合理的保守边界。
- 建议0–3用于router训练状态，4作为新登记的router开发核验；5–49锁定用于本轮扩大开发评测。
- 每模型450回合；BF16/A0/B4/B5四主配置1,800回合。若B5在B3/B4间选择，需内部B3固定控制，共2,250回合；主展示表仍可只列用户指定四项。task-only/random/shuffled控制另计，资源预算必须包含这些控制。
- 450较300增加50%回合，独立回合理想标准误仅缩小约18%；任务仍只有十个。要同时报告任务分层区间与配对rescue/break。
- 5–49已反复参与历史开发，不能称独立盲测；本轮不得再根据这些结果调router。
- 示范轨迹ID和reset索引不同。排除0–4不能排除PTQ校准暴露；需要训练、router、PTQ来源清单与实际初态/完整观测内容检查。历史封存留出标签仍不读取。
- 已完成内容审计覆盖A4开发20–39及后续B3/B4 20–49；不能宣称新5–19已通过。新片要单独核验。

## 6. 正式四模型对照的固定要求

BF16/A0/B4/B5统一checkpoint身份、task映射、reset5–49、processed首观测、norm stats、双相机、L1 head、8步chunk、夹爪处理、220步上限、稳定等待和种子协议。四者全部同条件实测；旧分母/旧参照不混入新主表。

正式片建议50+100+100+100+100=450/模型；微测使用首片的子集不额外加样本。首50后做协议硬审计，通过后按预注册顺序跑全部片，不能按成绩方向选择继续。GPU空闲但有合规待跑项立即恢复；无合规工作则分析备份关实例。资源起跑检查需估算全部trace、router训练数据和双份归档，留至少6GiB；历史磁盘读数不能当当前值。

## 7. B5的最小合理定义

B5-pilot-v1暂定义为：**A4共享基底上，冻结B3与B4整套adapter，用当前因果观测每8步Top-1选择一次。**

此为异容量策略选择诊断，不是同预算新专家SOTA。存储需要两套adapter；额外参数、router与切换成本全部报告。事前指定B5−B3与B5−B4两项比较，分别报告任务分层区间；只有稳定胜过两固定策略才支持超过最佳固定。另需task/instruction-only及随机/打乱上下文控制，防止把常选更强静态策略误认作状态路由收益。

300回合B3/B4有10rescue/8break，只是候选线索。事后并集293/300不是router可获得性能，不能拿评测成功标签直接训练selector。

现有H17是在语言0–17及视觉LoRA运行后取得，不能用于当前整套专家选择。应使用第一可选视觉层之前的独立图像/本体/指令特征；若额外运行无专家A4提特征，必须计算第二次前向成本。

router监督只来自训练reset0–3的完整学生访问观测，同一观测上冻结两专家与BF16教师查询；动作/夹爪差异是训练信号，闭环成功是主结果。已有80早期观测不足以覆盖完整选择过程，新增router数据单独登记；不改B3/B4原训练集。

B5尚无可运行训练/切换入口；本卡是具体前置设计。需补：固定E0/E1逐输出复现B3/B4、专家冻结、router白名单、保存/重载、8步唯一选择、特征在专家前、GPU10步、配对微测、首50trace硬审计。基线未完成不启动B5长实验。

