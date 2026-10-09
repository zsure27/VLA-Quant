# B4：全合格 W2 底座的视觉＋语言联合 LoRA

日期：2026-10-09。用户已授权下次自行开机后执行。**本轮只准备代码、CPU契约与预注册；没有连接服务器、运行GPU或新增成功率。** 当前真实GPU烟雾未通过；046完整本机归档尚待补传校验。

## 当前研究状态和证据

PRIMARY=P2 shared static PEFT recovery。主问题：在A4上，相同训练观测、教师、语言初始化与1000步预算下，扩大为视觉＋语言联合可训练范围，相对B3是否得到有价值的闭环增量？P4/P5/Router、静态量化搜索和offline_final_holdout继续锁定。

以[046累计分析](../reports/experiments/p2-shared-peft/20261008-046-b2-b3/DEV_BOUNDARY_30_39_AND_COMBINED_ANALYSIS_CN.md)为最新开发证据：reset20–39每配置200回合，A3=164、B1=194、B2=190、A4=30、B3=192、BF16=194、A0=195。B2−B1为3 rescue/7 break；B3−A4为162 rescue/0 break。旧独立seed复现无效，不能混入样本。

B3失败8回合，其中7回合BF16成功；B3还在5个BF16失败回合成功。因此相差2个成功**并不意味着只有2个可修复失败**。B4须同时检查救回这些失败和保留B3已有成功。B2是在A3上冻结B1后仅训练视觉，不能预测A4上联合训练的结果。

## 固定的B4定义

| 项目 | 预注册内容 |
|---|---|
| 底座 | A4；422个合格AWQ目标W2，语言/DINO G64、SigLIP G128，保护模块维持高精度 |
| 语言LoRA | 全32 blocks ×7类Linear，共224个，rank8 |
| 视觉LoRA | DINO与SigLIP共196个transformer Linear，rank8；不覆盖两处patch卷积 |
| 训练组织 | 420个Linear各有独立两矩阵，共840矩阵；一次最终动作损失共同反向传播，两种模态都更新 |
| 可训练参数 | 26,710,528；比B3多6,722,048，约33.6%；BF16因子裸张量约53.42MB，实际文件大小待测 |
| 初始化 | 精确复用B3的**训练前**Response-SVD语言因子；视觉标准零输出；不从训练后的B3继续优化 |
| 训练 | 同一80观测和教师标签；rank8、alpha/rank=1、AdamW lr1e-4/wd0.01、Smooth-L1 beta0.1、batch1、1000步、样本顺序seed7、全体梯度裁剪1 |
| 数据 | A4训练reset0–3，40学生回合前两次查询=80观测；明确policy_normalized_proprio，只归一化一次 |
| 主对照 | 已冻结B3；B3、B4、A4、BF16、A0在当前同条件重新评测，原200仅作回归参照，不追加独立样本 |

这是扩大可训练范围、同时增加容量与计算的工程总效果；不能称同参数预算下的联合训练优越性，也不能拆分归因视觉本身、联合优化或容量。视觉与语言因子没有跨层共享权重；一个文件封装不等于一对矩阵覆盖所有层。

不新做SVD、不新采集学生状态、不改变量化profile、rank、损失或teacher。B4接收两模态的联合state文件，单独注册在`configs/model_registry_b4_v1.json`。原B3 artifact绑定的训练器、评测器和registry全部保留原始字节，避免改代码后失去旧对照的可装载性。

## 下次开机顺序：必须先结束旧收尾

1. 只在用户明确当前实例已开机后，绑定其端口、专用密钥、隔离known_hosts和实际hostname。读取当前五小时实时额度，同步并编译核验收尾helper；不默认复用046身份，不自动开机。
2. 定位旧服务器归档`/root/autodl-tmp/qvla-repro/backups/b2b3-046-final-20261009-0001`，核验13项与18494项清单。先补传完整归档到本机忽略目录并逐文件校验；部分tar和缺清单目录不能当完整副本。旧046关机是用户报告，不补造原生回执或平台OFF核验。详见[旧收尾记录](../reports/experiments/p2-shared-peft/20261008-046-b2-b3/CLOSURE_20261009_CN.md)。
3. 服务器运行`record_prior_b2b3_closure.py --server --archive <旧归档> --expected-hostname <当前核验值> --output <新server-verification.json>`；取回核验文件。Windows运行既有`verify_vla_archive_local.py --archive <完整本机归档>`。完善旧收尾报告，使用`vla_push_local.ps1`推送并核验zsure27身份与远端SHA。
4. 本机运行`record_prior_b2b3_closure.py --archive <完整本机归档> --server-verification <取回核验文件> --output <新prior-closure-receipt.json>`，再次核验全部文件和GitHub main。该回执绑定两份清单SHA及核验数、当前远端提交。把回执复制到当前服务器，纳入不可变物料锁。**没有PASS_PRIOR_CLOSURE，B4注册及训练都会拒绝。** 最终上传旧收尾/回执的小型记录；大文件不进普通Git。
5. 资源门禁：起跑前持久盘空闲至少12GiB，其中6GiB为归档/异常余量；每片前按实际产物速度复核。旧046仅余约3GB，克隆本身不会扩容。需要扩容或保留完整已验证副本后整理可再生缓存；不能删唯一结果。旧SVD spool约9.5GiB可再生，但本轮准备不自动删除，必须先核验路径与备份/恢复来源。
6. 依据模板填写当前`materials`，定位B3的train1000、student-state80、profile、checkpoint和原始peft_train校准/切分。重新查全部SHA和数据角色；不把“已克隆”当通过。
7. 在独立checkout运行`prepare_b4_joint_plans.py --materials <json> --session <新持久盘会话> --output <新control/plans> --python <冻结overlay Python> --expected-hostname <当前核验值>`。它只注册三片，不启动GPU。保存计划/代码/物料hash并推送预注册快照。
8. 后台顺序runner启动`B4-first50-plan.json`。启用仅本轮活跃的约15分钟低频心跳；仅读status/revision、runner、GPU进程，未变化静默。阶段完成/失败/待跑而GPU空闲/额度门限才深入读取。

上述补传与核验阶段不同时启动B4。若归档缺失、物料不匹配、资源不足或需要用户审阅的问题当前无法解决，清楚记录问题并按约定备份关当前实例，不能付费等待。

## 实验顺序和硬门禁

- 真实10步烟雾：核验语言224＋视觉196的形状和W2量化；joint全零与A4动作完全等价；视觉零输出＋语言初始化与B3训练前动作等价；DINO、SigLIP、语言所有应训练矩阵梯度有限；主干/head/proprio等冻结参数和buffer SHA不变；重载动作完全相同。同步测峰值显存、时间。OOM或契约失败停止，不静默减rank、改初始化、改样本或改backbone。CPU契约不能放行长训练。
- 正式1000步从烟雾**之前**的联合初始化重新开始，不能累计1010步。教师cache与B3字节一致；烟雾中用冻结BF16再查询同观测，检查实际输入/标签是否重现。完整训练样本顺序SHA必须与B3一致。出现差异先诊断，不继续长测。
- 配对微测：五配置在reset20各10回合；核验实际argv、退出码、manifest、首个策略可见观测与完整动作/观测trace。微测只测协议，不作为10个额外样本。
- 第一片：reset20–24，五配置各50；协议硬审计后停AWAITING_GATE_REVIEW。同一心跳完成分析/实时额度检查后，协议通过才启动已注册next50。
- 第二片：reset25–29，各50；审计并分析，再决定是否按资源门禁启动已注册last100。
- 第三片：reset30–39，各100；完整严格配对汇总200/配置。**成功方向不参与协议放行**；成绩差也按固定计划完成，除非错误/磁盘/显存/时间/额度触发停止。资源截断时如实报告实际完成范围，不拿历史或重叠微测补足。

三份计划共用启动预算：含阶段间门禁最多6小时，单阶段最多3小时，最多一个B4候选。全程串行，无B2重训或新Router。训练时间根据10步实测更新估计；1000步预估分钟级至数十分钟，1000个正式评测回合为主要耗时，先按旧runner实际时间估算小时成本，再启动长片；这些是规划范围，不是GPU实测。

## 完整结果的固定分析

主指标B4−B3：200回合成功率差、rescue、break、双方成功/失败、任务分层、任务聚类95%配对区间与exact McNemar。辅助比较B4−A4、B4−BF16、B4−A0，同期B3参照也保留四格。列B3原8个失败及新B3/B4失败，不根据这些子集选择训练或另报无偏成功率。

分析实际失败轨迹的首次动作/夹爪/位姿分歧；不同策略的分歧后观测不能当同状态标签，需同观测教师查询时另立P2.5诊断。rescue/break并存只表示回合翻转，不能单独证明可路由冲突。记录参数、adapter字节、GPU时间、峰值显存及推理chunk成本；fake quant不能冒称packed INT2加速。

若净差为正而区间跨零，只保留“有方向的开发候选”；区间下界>0才支持当前开发分布上的增量，仍无独立泛化主张。若净差≤0、引入破坏或成本不抵收益，保留更简单B3作为默认，先解释剩余失败，停止自动增加scope/专家/rank。最终独立验证待方法冻结后另预注册；本轮不触碰封存数据。

## 收尾

当前五小时剩约15%停止扩展并完成原始数据＋分析的服务器持久盘、本机副本、同名reports/results及zsure27/VLA-Quant非交互推送/远端SHA；约10%停新GPU、原生关当前核验实例，至少预留3%。完整200分析完成可提前归档关机，不空跑。暂停本轮心跳。下一轮本地Windows关机须本轮新授权，旧许可不继承。

预注册卡：[PRE_RUN_CARD_B4_JOINT_CN.md](../reports/experiments/p2-shared-peft/20261009-b4-joint-preparation/PRE_RUN_CARD_B4_JOINT_CN.md)。代码审查和CPU证据保存在同会话reports/results。
