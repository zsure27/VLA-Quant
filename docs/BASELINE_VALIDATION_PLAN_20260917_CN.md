# AWQ / SmoothQuant 基线深入验证计划

日期：2026-09-17。用户已告知107开机，本轮已启动官方初态5–14的BF16/AWQ W4配对分片。最新优先级：AWQ基线实验优先，SQ和微调其次；由用户开机，Codex直接SSH，仍不自行登录或开机。

## 一、研究定位与已知证据

“跑通”分别验收：入口和数值正确、闭环任务有效、全量协议可比、真实低比特存储/执行。当前BF16承载的fake quant仅能支持精度与控制结论，不能宣称packed INT2/INT4加速或显存节省。

|路线|实测证据|目前能说什么|不能说什么|
|---|---|---|---|
|BF16 / AWQ W4A16|Spatial开发集各47/50，10任务×5初态|有限开发集存在可用W4参考|完整QVLA协议复现、所有任务无损、统计等价|
|视觉W2，语言BF16|Spatial 10/10|当前小样本视觉可耐受|视觉2bit普遍无损；DINO/SigLIP是两编码器，不是两相机|
|语言W2无额外clip，视觉BF16|1/10；32帧动作MSE0.06972661|取消clip缓解离线误差，但未恢复任务能力|无clip足以获得高成功率|
|全目标W2|0/10|当前组合明显失效|所有2bit方法必然不可行|
|语言8–23W4、其他语言W2、视觉BF16|48/50|混合精度中段保护可恢复开发表现|均匀W2成功或PEFT训练收益|
|SQ W4A4|旧环境0/20，不可与当前协议直接合并|当前基线尚未验收|当前实现已复现论文失败率|
|SQ W16A16数值控制|视觉分支及FP32成对控制仍超旧max MSE 1e-4筛查门槛；局部FP32等价误差约1e-14|存在整体动作扰动，需定位|超门槛必是逻辑bug；单组max误差能确定低比特失败根因|
|响应SVD初始化|rank8 MSE0.06361106，rank16 0.06399096；零训练步、无闭环|动作输入方向选择略改善开发误差|已恢复W2或已完成PEFT|

原始证据与图表：[9/16后续实验](../reports/sessions/20260916-107-followup/README_CN.md)、[9/17 SVD](../reports/sessions/20260917-107-response-svd/README_CN.md)、[早期路线图](AWQ_DIAGNOSTIC_ROADMAP_20260915_CN.md)。重复开发初态与帧不增加独立样本量。

## 二、先冻结共同评测契约（G0）

1. 模型checkpoint、processor、norm_stats、action head/proprio等来源hash；OFT双向注意力实现、双相机、center crop、8步chunk、7D连续动作、夹爪处理、步数上限固定。
2. 校准集来自训练轨迹，与最终评测划分分离；记录轨迹及取帧清单hash。冻结profile、搜索目标、G大小、clip规则、对称/非对称、zero point、scale dtype和AWQ坐标。
3. 固定LIBERO版本、任务列表、初始状态文件及hash、环境种子和模型种子。论文/官方入口语义与本地paired seed协议分别说明；不能拿不同seed协议直接与论文成功率作严格数值复现断言。
4. BF16原始入口与量化入口完全旁路对照；完全不施加scale/clip/quant时应复现同入口输出和成功。保护模块hash、目标覆盖统计及整数码域检查。
5. 为完整评测准备有限分片和恢复：每片以task ID、官方init index、seed和配置hash唯一定位；不能把num_trials减小当成“后续状态分片”，避免反复从初态0重跑。先实现并核验范围/恢复支持，再开长测。
6. 记录程序异常、依赖/环境失败与正常完成但策略失败的区别。异常episode仍按预先固定规则进入主评测分母，并单独公开基础设施异常数；不选择性删除失败。必要修复后完整重跑对应分片，保留旧日志。

此前50是rollout/episode，不是训练epoch。公开OFT入口默认50 trials/task；以Spatial 10任务×50=500作为完整suite最低安排。QVLA报告四suite（Spatial/Object/Goal/Long），使用对应有来源的suite checkpoint、统计文件及校准profile，计划每种配置合计2000 episode。公开默认参数不等于已证明论文表格使用的重复seed次数；这一点需继续核实，不能自行补造。

## 三、AWQ W4A16完整验收（W4）

首要任务：冻结一个W4 profile，在完整Spatial上与BF16配对各500；采用有限分片跨会话累计。旧50仅当所有配置/seed/源版本hash一致才可计入正式500，并保留“已用于开发”的标记；否则重跑，不拼接旧成绩。

输出：逐任务成功率、累计成功数、失败类型、配对成功/失败矩阵、W4−BF16百分点差；按task分层给出不确定性，并明确固定任务推广限制。若要宣称可接受掉点或非劣，正式跑前冻结允许损失阈值（例如2个百分点只是候选标准，尚未授权为最终研究结论）；不能用不显著差异证明等价。

目标不是要求W4与BF16必然一样好。QVLA表2的OFT Spatial BF16为97.6%，AWQ W4A16为93.0%，本来有性能损失。先与我们相同条件的BF16比较，再列论文参考值与协议差异。若我们BF16偏低，先修共同基线，不能一概归咎量化。

Spatial通过后扩展Object、Goal、Long各suite500，先检查各suite BF16可达表现及checkpoint能力，再独立量化；不能拿Spatial专用checkpoint直接代替其他suite对应模型。W4达到可靠参考后允许少量PEFTpilot，最终论文主表仍需四suite。

## 四、视觉W2扩大验证（V2）

第一阶段固定BF16语言：BF16全模型、双视觉W4、双视觉W2（当前DINO G64、SigLIP G128且保留clip）在与W4相同的Spatial500初态上配对；BF16参考只在完整契约一致时复用。

开发诊断可先各50/100，完整主要候选仍补至500；新50状态不得与旧10重叠计数。记录分任务掉点、抓取/释放阶段、位置/旋转/夹爪响应及失败回放。若双视觉W2退化，先用DINO-only / SigLIP-only W2在开发集定位，再扩大必要候选，不把所有组合都跑四suite。

与G128/G128比较属于组大小且重搜索profile的独立干预，额外记录元数据和scale/zero point开销。视觉仅W2成功只能表述为“视觉W2+语言BF16模型在该协议下有效”，不支持全模型均匀2bit、视觉永远无损或参数原封不动等断言。

## 五、语言W2无clip及PTQ恢复（L2）

先在相同Spatial50/100固定初态，比较原clip与移除语言额外clip，其他因素固定、视觉BF16；保留原clip失败对照，整数码域clamp仍保留。无clip严重失败时不把所有失败候选硬扩至四suite；正式进入完整500的候选需先在开发集体现稳定任务能力。

离线审计与闭环并行：层输出/动作token hidden的逐层误差、同教师输入与真实量化前缀输入、方向性误差、夹爪margin；后续仅对预选敏感阶段干预，并生成配对可视化。

优先级顺序：

- L2-A：G128/G64/G32新搜索，对相同校准预算比较clip保留/取消/有限重搜索，不能仅改旧profile的标签。
- L2-B：教师输入校准 vs 顺序量化前缀输入校准，检验分布偏移；保留种子、数据、目标和codebook。
- L2-C：同AWQ坐标下的少量W4/BF16 rescue与已测8–23混精参照，统计真实参数加权平均位宽以及元数据开销；混精单独命名。
- L2-D：确认原OFT base与LoRA来源后，检查先合并再量化是否吞掉adapter贡献；不能将Wmerged−delta当作未核实的原base。
- L2-E（辅线）：少量rank8静态补偿、scale-PEFT及动作蒸馏可行性。冻结主干并核验梯度、保存恢复；本轮既有SVD无需继续无目标加rank。

最终评价以闭环任务能力为主。论文没有AWQ均匀W2基线条目，均匀W2是本项目更激进的新研究条件；允许严谨否证“无clip能恢复”，不能强迫成功。研究出口是可解释的PTQ改进、注明预算的混精或PEFT补偿；只有有任务收益才扩至500及其他suite。

## 六、SQ W4A4修复与同协议验收（SQ）

目标是查明当前实现准确性并得到可靠W4A4成绩，不能先保证必然达到论文77.2%或BF16水平。QVLA表1的OFT SQ W4A4 Spatial77.2%、四suite平均73.4%，显著低于原BF1697.6%/97.1%。SQ W4A4基线与QVLA自身平均4bit方法不同，不能互相替代。

SQ0实现/论文差异清单：官方baseline源码/commit和运行参数、alpha、校准训练轨迹与指令混合、实际W/A量化目标及排除模块、signed范围、per-row/per-token语义、QK/PV是否量化、Conv输入的像素/patch量化轴、norm affine及bias缩放、共享QKV/MLP输入配对、hook是否重复或在错误缩放坐标上量化。公开QVLA README主要提供QVLA gate/fake-weight流程，未给出表格AWQ/SQ完整baseline命令；该README缺失不能证明整个仓库没有baseline，尚需核实真实可得实现，缺少信息明确列未确认，不擅自称严格复现。

SQ1数值控制：BF16重复；scale-only W16A16；语言only、两视觉及单视觉；局部全FP32与完整视觉FP32成对教师；逐模块查找首个偏差、bias/axis/残差支路错误及BF16舍入累计。完整视觉FP32是诊断支路，不可冒充最终BF16 SQ baseline。已经测过的局部/成对FP32不盲目重复；完整视觉FP32控制仍未执行。

原max动作MSE1e-4是筛查门槛，不是数学正确性的充分或必要条件。局部FP32代数、BF16旁路噪声、端到端配对闭环共同判断；若确定只是有限精度舍入且闭环影响可接受，记录证据和阈值依据后可以继续A4，不能为过门槛随意放宽阈值，也不能无限追逐max MSE掩盖真正的A4瓶颈。

SQ2权重/激活定位：在相同已核验平滑图下比较W16A16、W4A16、W16A4、W4A8、W8A8、W4A4。重点测视觉/语言/动作token的激活absmax、分位数、饱和率、有效码利用率、逐层误差。各组合误差不可相减当作独立因果贡献。

SQ3修复候选：优先修axis、hook、配对与坐标错误；若代数正确而A4失效，调整alpha及校准覆盖、比较token/组量化颗粒度。局部A8 rescue仅为诊断或混精方案，不能标成纯W4A4成功。每一种改动单独编号、保持数据/预算，分别标“论文协议候选”“VLA改进候选”。

SQ4正式评测：数值/实现契约明确后先50/100配对Spatial，再完整500及四suite。固定BF16、同checkpoint、相机、chunk、归一化、seed、初态与运行时源码。若仍明显偏离参考，继续报告协议差异和失败证据，不能删日志、换初态凑成功率。

## 七、执行顺序与输出

本轮顺序：G0及分片核验 → BF16/W4完整Spatial有限分片 → V2扩大配对 → L2-A/B归因 → W4跨suite准备 → SQ数值定位和验收。短预算优先选择能完成的AWQ诊断，不因预算短而把SQ排在AWQ前面。微调安排在AWQ基线里程碑之后，Router最后。CPU源码审计可在AWQ GPU实验运行时记录，但不挤占AWQ测试。

每片持久化逐episode JSONL/CSV：suite/task/init index+hash/seed/config hash/success/episode error/steps/time；分任务柱状图、配对矩阵、累积率和失败视频；诊断另存逐帧动作/hidden/activation数值与分析。视频可保存在双份大文件备份，报告链接对应清单hash；未测格空白，不为计划生成假成功曲线。

每轮结果单独reports/sessions/YYYYMMDD-SESSION/，原始小日志results/SESSION/互引，分析及source hash随正确账号zsure27/VLA-Quant同步，保留失败候选。

只看五小时剩余额度：>=20%每3短/2长检测，10%–20%每实验后检测，<10%立即备份关机并至少预留3%。分片不能跨越安全收尾预算；读数脚本须先落实仅五小时判定，周窗口缺失不得阻止实验。检测不可用按既定快速恢复后收尾。不设置定时任务；由用户自行开机告知实例，Codex直接SSH。

## 来源和协议边界

- [QVLA论文v1，表1/2与附录F](https://arxiv.org/html/2602.03782v1)：四suite与参考成绩；512校准轨迹是QVLA敏感性分析配置，不能推定所有AWQ/SQ baseline也用了相同512设置。
- [QVLA公开OFT评测入口](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/main/openvla-oft/experiments/robot/libero/run_libero_eval.py)：默认每任务50次、官方初态索引；具体表格重复seed次数未由默认参数确认。
- [QVLA公开README](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/main/README.md)：gate与fake-weight流程说明，不能替代表格baseline完整参数。
