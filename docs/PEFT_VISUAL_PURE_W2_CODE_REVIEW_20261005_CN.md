# 视觉增量 LoRA 与全合格量化层 W2+LoRA：代码审查（2026-10-05）

## 审查结论

**两张 PRE-RUN CARD 已写定；当前代码不能直接执行其中任一项长训练/长评测。** 本轮仅做仓库静态审查与既有产物复核；059 已收尾，没有在新实例做加载、训练或 GPU 烟雾。下次开机可以直接从下面的实现与烟雾门禁开始，不能把本审查误记为运行门禁 PASS。

本次审查的关键源码 SHA256：`diagnostics/probe.py`=`e42882dd4e80bd2964ec0b3922d5494a4338d8cc7a42c8752be470894ac29777`；`qvla/run_eval_official_quant.py`=`aaf92ce5257d120437ef9830a0ed57341f8714a83755dfdabebd77c1a38991cf`；`diagnostics/low_rank_recovery.py`=`8cd38967641c0c7bacf6ba3cca859fb55f471b0627dbe0db73bfb39306f99cd7`；`qvla/recovery_lora.py`=`2fabd32749f52e7b269e3fe66d0e939b2db17a004e1bc4b504e71191de5891f9`；12L 配置=`a30d68788bc2aa101ffa4f46d76def7d93b4402fd293db704a305cf254d2d35c`。下次若源码不同，重审受影响路径。

| 严重度 | 已核实的代码路径 | 对实验的影响 | 放行前需要的修复及可验收证据 |
| --- | --- | --- | --- |
| 阻断 | `diagnostics/probe.py` 的 `--awq-residual-layers` 只解析语言 block；`selected_names` 与 `residual_selected` 均要求 `language_model.`（约 543–555、783–785、846–859 行） | 视觉目标即使在 AWQ plan 中也不会被训练为 LoRA，现有命令改参数无法得到视觉增量 | 新建版本化视觉 target whitelist，限定已量化 `Linear`，打印并哈希实际目标/shape；证实仅视觉新参数有梯度，LW 和所有量化权重冻结；10 步训练和零输出/重载检查通过 |
| 阻断 | `qvla/run_eval_official_quant.py:125–132` 只允许一个 PEFT state，且钉死 exact-12L candidate | LW+Vision 需要在冻结 LW 基础上再加载视觉参数；现有 evaluator 只接受单份语言 state | 增加显式、相互不重叠的语言和视觉 state 入口，逐 key 校验、合并后精确 attach；保留旧 12L 路径的严格门禁与回归结果 |
| 阻断 | `qvla/run_eval_official_quant.py:159–205` 的 LoRA key 正则只允许 12L 仍为 W2 的 20 个语言 block、每 block 7 个 Linear；`333–355` 要求所有 state key 确实 attach | 视觉 key 会被拒绝；纯 W2 新变成 W2 的 12 个旧 W4 block 也会被拒绝 | 对两个新版本分别规定精确目标 schema；从**实际** AWQ plan 生成 expected keys，拒绝多余/缺失、shape 不符和非有限值；保存/加载后比较数值和有效作用范围 |
| 阻断 | `qvla/run_eval_official_quant.py:127–132, 489–497` 要求 W4 blocks8–15/20–23 的组合路径 | 不能把删掉 W4 layers 的参数硬塞到现有 PEFT candidate；纯 W2 也无可审核的独立 backbone 配置 | 新增版本化全合格层 W2 配置与 evaluator candidate；真实加载后导出每层位宽/group/clip/profile SHA；保护模块单列高精度，不把混精旧 adapter 迁移结果冒充新训练 |
| 高 | `diagnostics/probe.py:562–565, 779–824` 的 Response-SVD 校准要求 action-token，且收集器只选语言模块 | 语言校准帧与视觉 token 的语义不同，不能复用为视觉初始化 | 视觉增量先采用零输出初始化；若以后测视觉 Response-SVD，单独预注册视觉 token 采样与轨迹来源 |
| 高 | 现有 `scripts/run_20260930_059_language_w2_all_v3.sh` 与训练烟雾验证器固定 059 路径和语言 20-block 目标 | 修改目标字符串后不能视为新实验脚本；克隆机路径/hostname/输入可能不同 | 两实验各自建立不可变计划和入口，按新实例重新核对 SHA、目标数、退出码及恢复标记；不用旧 059 文件名冒充新方案 |
| 高 | 2026-09-30 首片曾因实际命令缺少 `--trace-observations` 被判协议失败 | “命令看起来配对”不足以审计首观测和轨迹 | 新计划的微测和首片检查**实际执行命令**、trace 文件、manifest、初态 SHA、8 步 chunk 与非有限动作；首片最多 50 回合/配置并在 gate 停住 |
| 高 | 110 A1 的 `env_seed` 字段变化曾在真实 seed 调用顺序下不形成独立条件 | 下一次长评测若只是换数字，会重复无效样本 | 严格执行 `docs/EVALUATION_SMOKE_GATE_20260929_CN.md`：真实 evaluator 顺序、无模型同条件重复、策略可见处理后输入、配对微测；开发 reset 与独立条件分别报告 |

## 视觉增量特有的设计风险

`diagnostics/low_rank_recovery.py` 的通用 residual 附着以 `Linear` 为前提。视觉 AWQ 目标不等于全部可训练 LoRA 目标；必须按真实模型类型枚举，排除卷积和保护模块。冻结 LW 后新增视觉 adapter 是一个清晰的工程干预，但会增加参数预算；它不能证明“视觉比语言更有效”。并且旧 LW 卡的视觉确认性门禁未通过：LW−C3 为 24/300，任务聚类区间下界约 −0.3 个百分点。按新卡只能做清楚标记的探索性烟雾/微测，不能事后称旧门禁 PASS。

## 全 W2 特有的设计风险

新底座先要证明每个合格 AWQ 目标均为 W2，以及 DINO/SigLIP group、语言 G64 和 clip 实际值。保护模块维持高精度，所以名称应为“全合格量化层 W2A16”。主对照必须是**同一新底座**的 W2 无 LoRA；12L+LW 是参照。以 12L 学生状态训练纯 W2 会引入状态分布错配；新卡要求纯 W2 自身访问的 40 回合/80 query 及同观测 BF16 标签。若其动作不稳定到无法采集有效状态，先作失败诊断，不把旧样本静默替换进去。

## 下一次开机的可执行放行序列

1. 核验当前实例 hostname/GPU、服务器持久盘与已归档 059 物料；恢复 closure-tools，核对脚本与模型/profile/adapter/初态 SHA。
2. 分别完成两条路径的代码实现与静态/小 GPU 契约：目标清单、量化前后形状、保护模块、零残差、state key/shape/有限性、保存重载、冻结性、10 步梯度。两卡各出独立 `PASS_CODE_AND_10STEP_SMOKE` 或 FAIL 记录，不能互相借用。
3. 执行真实 evaluator 顺序的独立条件检查；若新条件不可得，明确仅做复用官方 reset 的开发微测，不宣称独立复现。
4. 每卡先各自配对微测，再首片最多 50 回合/配置，审计协议通过才启动预注册的后续片。不得根据首片成功率临时换 reset、训练数据或 adapter。
5. 每个阶段由服务器可恢复 runner 自动接续，活跃期间约 15 分钟低频读 status/runner/GPU；五小时实时额度约 15% 开始归档/Git 同步，约 10% 停新实验并关当前实例，至少保留 3%。无合规工作时提前分析备份关机。

相关预注册：[视觉增量](pre-run-cards/VISUAL_INCREMENTAL_LORA_20261005_CN.md)、[全 W2](pre-run-cards/PURE_W2_RECOVERY_LORA_20261005_CN.md)、[数据划分](PEFT_DATA_SPLIT_AUDIT_20261005_CN.md)。
