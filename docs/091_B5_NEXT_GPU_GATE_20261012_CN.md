# 091 下次开卡执行卡

## 已完成与仍未解决

CPU可完成部分已结束：历史500/450逐回合口径、实际W4 profile与checkpoint内容身份、422目标形状与分散行的整数码/重建核验、部署版本及21项CPU契约。B5代码、训练/评测入口和独立不可变计划已注册，正式router参数和GPU结果尚不存在。

QVLA表2 Spatial为BF16 97.6%、AWQ W4 93.0%；本项目历史500为97.4%和97.2%。92.5%是论文四suite均值。仍未实测解释W4的4.2个百分点差异：作者完整AWQ校准与视觉适配配方未公开，当前语言官方AWQ＋视觉自写适配不能称严格论文复现。不得为了接近93%调差基线。

091实际旧overlay中`env.seed`后调用`seed_all`，覆盖环境随机种子；已保留旧文件并独立部署修正。真实连续reset/processor探针在2GiB无卡环境被系统Killed，软件渲染已补齐，但实际输入门禁未通过。CPU参数检查不能替代真实输入、完整GPU前向或成功率归因。

## 执行来源

- 代码Git：`eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d`。
- 独立来源：`/root/autodl-tmp/qvla-repro/control/b5-pilot-091-20261012-v2/source`。
- 最终计划：`/root/autodl-tmp/qvla-repro/control/b5-pilot-091-20261012-v2/plans/`；不运行旧v1计划。
- 修正OFT：`/root/autodl-tmp/qvla-repro/control/cpu-audit-20261012/oft-corrected-v1`；评测器SHA `a4061279eebe82944f048bebfc9e85fa85acf0d7a9cf5d6ce18aea1102694fe2`。
- 正确Python：`/root/miniconda3/envs/qvla-oft/bin/python`。
- 当前身份：091端口18575，hostname `autodl-container-dfbc4bb0ff-1e85dd94`。用户重新开机后仍须确认实际身份与SSH密钥，不自动开卡或登录控制台。

## 顺序门禁

1. 用户启用GPU，并提供足够主机RAM；起跑持久盘空闲至少12GiB，运行保留6GiB。本轮清理经双份核验的重复导出包后实测10779123712字节，约10.0GiB，尚不足。扩容或先审阅可再生缓存，不能删除唯一数据。
2. 运行`baseline-and-bank-smoke-plan.json`：当前资源检查→真实协议观测探针→同条件BF16/A0配对微测→各50基线审计→B3/B4 query10 bank/static严格等价与重载。任何失败均停止本片，保存失败数据后诊断，修复必须新提交、新目录、新计划。
3. 若需要随机流归因，独立`baseline-bridge-plans/baseline-bridge-plan.json`在相同reset0–4对照paired与upstream流程。它是项目wrapper下的协议诊断，不是作者严格复现；先分析命令、输入哈希、配对rescue/break，不能按成功率方向放行。
4. 基线与bank门禁通过后，`capture-label-fit-plan.json`采完整回合、标注、训练router；再micro、首50硬审计，随后10–19、20–29、30–39、40–49分片。跨片验证前片gate、实时额度及磁盘，不重跑完成项，不让待审查GPU空跑超过一次心跳。

## B5问题与比较

冻结A4底座，Top-1选择冻结B3语言专家或B4语言＋视觉专家。前向前的特征仅取当前同策略crop双图像、一次归一化本体及指令，每8动作选择一次。禁止post-adapter H17、教师特征和未来成败。

router训练reset0–3，开发reset4，评测reset5–49，每模型450回合。保留BF16/A0/B3/B4/B5、instruction-only控制及首50 random诊断；正式总预算2750回合，微测不相加。主表BF16/A0/B4/B5，同时报告B5−B3和B5−B4，不能隐去更强固定专家。

数据仍为同十任务的新初态开发验证，非盲测或未见任务；offline_final_holdout不读标签、不改角色。B3/B4 bank异容量且总计46,699,008参数，须报告全部bank字节和延迟，不能宣称同预算专家创新。AWQ当前为BF16存储上的伪量化数值实验，不能声称真实INT2部署收益。

## 本轮保存

046旧完整归档本机逐文件SHA通过：13＋18,494项。091旧完整归档通过：8,952＋8,897项。原始实验和可再生SVD spool仍保留在服务器；本机恢复到标准被忽略备份目录。当前CPU结果与最终计划采用同名reports/results并同步GitHub；无卡无法完成的真实输入与GPU门禁明确保留为待项。
