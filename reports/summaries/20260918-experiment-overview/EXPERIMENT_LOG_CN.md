# 实验日志：按实验名称组织

> **2026-09-25 增量结果与执行覆盖：** 完整 W2/W4 混合底座在同一 Spatial500 上得到 16L 434/500、14L 431/500、12L 412/500；完整证据见[2026-09-24 至 25 会话报告](../../experiments/p3-static-backbone/20260924-107-awq-stage-complete/README_CN.md)。静态 block 搜索现已冻结，12L 固定为 blocks18–19 shared PEFT 主底座；后续因果顺序见[主线执行协议](../../../docs/PEFT_CONTEXTUAL_ROUTING_EXECUTION_20260925_CN.md)。本日志中较早的“下一步”仅保留历史语境。

更新：2026-09-18；覆盖已提交且本地可核对的实验，最新实测截止北京时间03:37。本文件为索引化日志：记录配置、实际结果、解释与原始证据入口，完整console/manifest/命令仍保留在链接的`results/`。拒绝或失败启动不计作完成实验；重复控制不增加样本量。

## E01｜AWQ-W4基线验证

**目标：**确认至少一条量化基线可继续做恢复方法对照。OpenVLA-OFT Spatial，AWQ W4A16 G128；官方初态0–49，10任务，每配置500回合，冻结checkpoint/evaluator/profile及配对种子。2026-09-17分片完成。

**结果：**BF16 487/500，W4 486/500；共同成功477、共同失败4、BF16-only10、W4-only9；500条manifest一致，0异常、退出0，13,426查询finite。旧开发0–4的50为47/50与47/50，不直接拼入正式数据。

**解释：**当前Spatial对照成立；不证明统计等价、四suite完整复现或真实INT4效率。

证据：[正式500报告](../../experiments/p0-foundation-baselines/20260917-107-baseline-continuation/README_CN.md)、[原始分片](../../../results/experiments/p0-foundation-baselines/20260917-107-baseline-continuation/)、[较早220回合分片](../../../results/experiments/p0-foundation-baselines/20260917-107-baseline-validation/)、[早期基线gate](../../../results/experiments/p0-foundation-baselines/20260916-awq-baseline-gate/)。

## E02｜AWQ-W2语言/视觉粗分区闭环诊断

**配置：**历史同官方初态index0的10任务，配对hash/seed。BF16、全W4、全W2、仅语言W2/no-clip、仅视觉W2 D64/S128。

**结果：**分别9/10、8/10、0/10、1/10、10/10；原语言clip为0/10。作为早期筛选，不能相加进后续500，更不能以视觉10/10证明2bit普遍无损。后续E13已修正该判断。

证据：[历史闭环表](../../archive/20260916-weekend-review/data/closed_loop.csv)、[早期可视化诊断](../../../results/experiments/p0-foundation-baselines/20260915-awq-visual-diagnostics/)、[来源与边界](../../archive/20260916-weekend-review/README_CN.md)。

## E03｜AWQ-W2语言额外剪裁撤销：固定输入离线诊断

**配置：**语言W2/G128，其他模块BF16；保留原scale，只撤销clip_max；诊断32帧、既有验证32帧，每帧8步动作。

**结果：**诊断动作MSE原clip4.025737→全无clip0.056331；仅attention无clip0.167240、仅MLP无clip0.294206；夹爪149→10/256。既有验证历史摘要原4.110546→0.069727；W4为0.000272。

**解释：**撤销剪裁大幅减轻异常，不等于闭环恢复；表中指标是相对BF16教师。未复取历史逐帧值不伪造。

证据：[语言CSV及逐帧数据](../../archive/20260916-weekend-review/data/)、[来源摘要](../../archive/20260916-weekend-review/data/archival_offline.json)。

## E04｜AWQ-W2视觉分支、剪裁与scale组缩小

**配置：**历史语言W2/no-clip固定，比较视觉BF16、DINO/SigLIP W2及group。自定义逐Linear/Conv视觉适配；不是PEFT。

**结果：**DINO W2/G128动作MSE0.08953，取消clip恶化到0.10641；保留clip重校准G64为0.07754。双视觉改D64/S128后既有组合验证均值0.115294→0.102585，夹爪31→26/256，24帧改善8帧恶化，中位数略升；旧全W2闭环仍0/10。

**解释：**语言不剪裁规则不能照搬视觉；G64来自重新搜索，不是仅改标签。

证据：[视觉历史CSV](../../archive/20260916-weekend-review/data/vision_offline.csv)、[组合逐帧记录](../../archive/20260916-weekend-review/data/vision_combination_validation.csv)。

## E05｜AWQ-W2选定block局部审计

**配置：**选定3诊断帧、5blocks、教师/量化前缀两类输入、8变体，共240条局部审计；检查scale-only、clip与精度干预。

**结果：**scale-only action-token hidden相对MSE约1e-5；部分局部输入上保留clip优于无clip，中层attention/后层MLP有局部保护线索。

**解释：**难例定点审计，不是完整层敏感性排名；局部重构不能替代完整前向或闭环。

证据：[原始metrics和controls](../../../results/experiments/p0-foundation-baselines/20260916-awq-block-audit/)、[局部误差CSV](../../archive/20260916-weekend-review/data/selected_block_error.csv)。

## E06｜AWQ-W2语言中段回退W4恢复诊断

**时间：**2026-09-16，107。语言W2G128/no-clip，视觉BF16，完整8-block阶段换用自有W4 profile。

**结果：**32帧基线MSE0.06972661；0–7、8–15、16–23、24–31回退分别0.05122427、0.02968988、0.03583929、0.06314109。十初态8–15为7/10，8–23为9/10。后续8–23扩大0–4初态开发50为48/50，旧BF16/W4均47/50；无episode error、manifest匹配。

**解释：**中段精度容量可恢复策略；半数语言blocks W4、视觉BF16，不能报纯W2或PEFT。

证据：[阶段原始结果](../../../results/experiments/p0-foundation-baselines/20260916-107-diagnostics/)、[扩大50原始结果](../../../results/experiments/p0-foundation-baselines/20260916-107-followup/)、[阶段逐帧/动作维度CSV](../../experiments/p0-foundation-baselines/20260916-107-diagnostics/data/)、[50回合CSV](../../experiments/p0-foundation-baselines/20260916-107-followup/data/closed_loop_50.csv)。

## E07｜AWQ-W2固定缩放坐标的模块族/位宽干预

**配置：**不借W4 profile，保持原W2 scale/no-clip，仅8–15的attention、MLP或全部Linear以4bit模拟。

**结果：**attention MSE0.04736661、夹爪9/256；MLP0.04872760、10/256；全段0.02988279、4/256。全段十初态8/10，自有W4方案7/10。

**解释：**提高精度本身解释大部分恢复；attention与MLP都存在损伤。模块数量不是相同参数预算，1个成功差不证明优越。

证据：[模块族CSV](../../experiments/p0-foundation-baselines/20260916-107-diagnostics/data/family_summary.csv)、[固定坐标CSV](../../experiments/p0-foundation-baselines/20260916-107-followup/data/awq_summary.csv)、[对应原始metrics](../../../results/experiments/p0-foundation-baselines/20260916-107-followup/)。

## E08｜AWQ-W2权重残差SVD低秩初始化容量测试

**配置：**107，语言8–15的56Linear添加BF16 BA分支；rank4/8，其他语言W2/no-clip；固定seed、training_steps=0。

**结果：**rank4参数2,498,560，MSE0.06740638，夹爪12/256；rank8参数4,997,120，MSE0.06723416，13/256。基线0.06972661，改善约3.3%/3.6%。未解释权重残差比例均约99%。

**解释：**权重残差低秩可恢复有限，不是训练效果、不是闭环收益。增加rank未稳定改善夹爪。

证据：[原始测量](../../../results/experiments/p0-foundation-baselines/20260916-107-followup/)、[低秩汇总与逐帧](../../experiments/p0-foundation-baselines/20260916-107-followup/data/low_rank_summary.csv)。

## E09｜AWQ-W2输入RMS加权SVD初始化

**配置：**同rank8/4,997,120参数，8个校准输入sample64–71，32个验证输入1000–1031；teacher缩放坐标，比较全token与真实动作token RMS。

**结果：**全token MSE0.06573510、夹爪14/256；动作token0.06544165、13/256；权重SVD参照0.06723416、13/256。training_steps=0。

**解释：**输入加权略好，但对角统计忽略通道相关性，仍未处理真实量化前缀；没有独立统计或闭环优势结论。

证据：[输入加权CSV](../../experiments/p0-foundation-baselines/20260916-107-followup/data/input_diag_summary.csv)、[原始metrics与校准manifest](../../../results/experiments/p0-foundation-baselines/20260916-107-followup/)。

## E10｜AWQ-W2动作token响应SVD及rank匹配对照

**时间：**2026-09-17，107。8校准输入，每模块448动作token行；32开发帧。响应ΔW·Xᵀ主方向初始化BA，rank8/16；另测rank16动作RMS。

**结果：**响应rank8 MSE0.06361106、夹爪11/256；响应rank16 0.06399096、13/256；动作RMS rank16 0.06503257、13/256。参数4,997,120/9,994,240。响应校准剩余能量均值15.85%/4.65%，不是整策略误差。

**解释：**比原W2改善约8.8%，比同rank8 RMS改善2.8%；局部响应恢复未转化为大的动作恢复，rank增加不单调。全部training_steps=0，无闭环LoRA实验。

证据：[本轮报告](../../experiments/p2-shared-peft/20260917-107-response-svd/README_CN.md)、[源数据](../../experiments/p2-shared-peft/20260917-107-response-svd/data/)、[原始结果](../../../results/experiments/p2-shared-peft/20260917-107-response-svd/)。

## E11｜AWQ-W2语言G128/G64×clip四格闭环

**时间：**2026-09-17。仅语言W2，视觉BF16；同官方初态5–9，每配置50，对应BF16 50。

**结果：**G128clip0/50、G128全无clip8/50、G64clip0/50、G64全无clip20/50；0异常、退出0。派生no-clip profile独立保存，不覆盖源profile。

**解释：**缩组与撤销剪裁组合有效但远未恢复；G128/G64搜索scale不同，不是单因素group试验。

证据：[本轮报告](../../experiments/p0-foundation-baselines/20260917-107-baseline-continuation/ANALYSIS_CN.md)、[原始语言分片](../../../results/experiments/p0-foundation-baselines/20260917-107-baseline-continuation/)、[最新合并scope数据](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/scope_paired_episodes.csv)。

## E12｜AWQ-W2注意力/MLP剪裁分离及扩展初态验证

**时间：**2026-09-18。G64源profile派生两份：仅撤销V/O64处clip或MLP96处；Q/K原本无clip，其他全部字段精确相等，源profile不修改。

**结果：**同5–9初态attention无clip29/50、MLP无clip0/50；结合原profile四格为[[0,0],[29,20]]。attention方案扩展10–19为49/100，对应BF16 96/100；0异常、退出0、完整manifest配对。

**解释：**attention剪裁是重要损伤因素，MLP保留clip有开发集优势；扩展仍明显落后BF16。不是PEFT，扩展不是全项目盲测。

证据：[profile逐字段核验](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/profile-provenance/)、[开发四格](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/language_clip_factorial.json)、[扩展分批数据](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/language_attention_splits.json)、[原始分片](../../../results/experiments/p0-foundation-baselines/20260918-107-awq-continuation/)。

## E13｜AWQ-W2视觉基线扩大至Spatial500

**时间：**2026-09-17至18。DINO64/SigLIP128，语言BF16；5–14、15–24、25–29、30–39、40–49、0–4六片覆盖500。

**结果：**230/250+94/100+91/100+45/50=460/500，对应BF16 487/500。BF16-only37、候选-only10、共同成功450、共同失败3；0异常、contract一致。task1 33/50对50/50；task5 40/50对47/50；task7 40/50对45/50。

**解释：**视觉2bit存在非均匀损失，修正早期10/10推断。task1初态7/8/14实际视频已查看，但没有独立逐阶段人工标签或固定输入归因。

证据：[500分析](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/ANALYSIS_CN.md)、[逐任务/逐回合CSV](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/)、[旧250原始日志](../../../results/experiments/p0-foundation-baselines/20260917-107-baseline-continuation/)、[新250原始日志](../../../results/experiments/p0-foundation-baselines/20260918-107-awq-continuation/)。

## E14｜AWQ完整W2原始配方与G64注意力改进配方

**时间：**2026-09-18。全部422既定连接目标W2，保护action head等；初态5–9，各50回合。

**结果：**原始G128 0/50；G64+注意力无clip11/50；BF16 50/50。均0异常、退出0、manifest匹配。

**解释：**完整策略仅部分恢复。新方案SigLIP也G64，与E13的D64/S128不同；改变group、搜索和clip，不能单因归因。下一步固定语言方案分离视觉分支。

证据：[最新scope汇总](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/scope_summary.json)、[full-w2两个原始目录](../../../results/experiments/p0-foundation-baselines/20260918-107-awq-continuation/)。

## E15｜SQ-W4A4历史参考与W16A16仅平滑控制

**配置：**旧环境低比特SQ成绩单独标历史；2026-09-16实际8帧控制W16A16，BF16教师，比较repeat、全平滑、关闭语言、projector教师替换。

**结果：**旧SQ W4A16 19/20、W4A8 20/20、W8A8 20/20、W4A4 0/20。当前repeat MSE0；全平滑均值0.0001543565/最大0.0010221635；无语言平滑0.0001023566/最大0.0005603094；projector替换后0。

**解释：**当前问题不是A4才出现；视觉数值变换可传播到动作。教师替换为oracle定位，不是修复方法。1e-4筛查未过，不报SQ验收成功。

证据：[历史SQ CSV](../../archive/20260916-weekend-review/data/historical_sq.csv)、[当前控制CSV](../../experiments/p0-foundation-baselines/20260916-107-diagnostics/data/sq_control_summary.csv)、[原始summary/metrics](../../../results/experiments/p0-foundation-baselines/20260916-107-diagnostics/)。

## E16｜SQ视觉分支与局部FP32/BF16平滑等价性

**配置：**语言不平滑，8帧只平滑DINO或SigLIP；sample65选DINO/SigLIP各3blocks的norm1→qkv/norm2→fc1共12组，三数值情形共36项。

**结果：**DINO动作MSE0.000123520/最大0.000807842；SigLIP0.000251068/最大0.001973883；均不满足最大≤1e-4。所选12组FP32近似等价，BF16有约1e-5量级局部相对误差，精确逐组值见CSV。

**解释：**两个分支均可产生差异；组合可能抵消。12组局部等价不证明全部162组或端到端正确。

证据：[分支与逐组CSV](../../experiments/p0-foundation-baselines/20260916-107-followup/data/)、[局部原始metrics](../../../results/experiments/p0-foundation-baselines/20260916-107-followup/)。

## E17｜SQ完整视觉配对组提高FP32的数值定位

**配置：**98组norm→qkv/fc1提高计算/参数FP32，但Linear输出返回BF16；两条8帧路径分别未平滑与平滑，第二条用第一条作教师。

**结果：**未平滑对原BF16平均5.54434e-5/最大2.28740e-4；平滑对同路径教师平均2.38356e-4/最大1.76225e-3，均未过原筛查。

**解释：**局部提高精度不足；参照不同不能直接比较收益。下一全视觉FP32控制曾被自动审批阻塞，命令未执行，不生成结果。

证据：[FP32汇总/逐帧](../../experiments/p0-foundation-baselines/20260916-107-followup/data/sq_fp32_pairs_summary.csv)、[原始FP32结果](../../../results/experiments/p0-foundation-baselines/20260916-107-followup/)。

## E18｜Contextual Routing专家互补性预检查

**配置：**读取同初态5–9实际结果，严格配对，两种语言W2候选组合；CPU分析，无路由训练。

**结果：**G128/G64全无clip共同8、G128独有0、G64独有12，oracle20/50等于最佳静态。G64全无clip/仅attention无clip共同15、前者独有5、后者独有14，oracle34/50、最佳29/50。

**解释：**存在一组正线索和一组负对照；oracle使用未来成功标签，不是部署结果，不作为同测试集训练标签。尚无同共享基座PEFT专家或扩展100互补性验证。

证据：[正互补JSON](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/language_all_attention_complementarity.json)、[负互补JSON](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/data/language_g128_g64_complementarity.json)。

## E19｜OFT adapter来源/量化范围审计

**配置：**CPU比对adapter safetensors形状与模块映射；含4D Conv1x1 B。439模块：语言225、视觉211、其他3；422量化目标连接原adapter。

**结果：**源shape映射通过；语言adapter参数约81.11M，视觉约28.697M，其他约1.019M。实际runtime merge数值等价未验证，训练步数0。

**解释：**文件映射不能证明可拆还原base；不执行merged−BA伪还原，不报高精度原LoRA保留效果。

证据：[OFT审计汇总](../../experiments/p0-foundation-baselines/20260917-107-baseline-continuation/data/oft_adapter_scope_summary.json)。

## 执行异常、数据完整性与未完成事项

- 107阶段诊断首启动缺`smoothing_selection.py`哈希依赖，退出后补齐并预检查，重新运行完成；失败日志保留，不计入指标。
- 自动审批曾引用上一轮关机记录拒绝全视觉FP32控制；未获准命令未执行。响应SVD在后续重新开机授权后完成，不能沿用旧“未执行”摘要覆盖最新结果。
- 上轮额度耗尽导致关机请求在自动审批阶段未执行，107持续在线。本轮已改为提前准备/15%收尾、单命令备份推送关机，实际回执另存。
- 2026-09-18新增550视频和两份完整派生profile双份归档、10项SHA校验通过；15份trace共19400查询finite。视频/NPZ/大profile不进普通Git，模型和校准原件仍不能称全部异地备份。
- 未完成：认证当前SQ W4A4同协议闭环、实际梯度PEFT训练及闭环、可部署Router、完整四suite/多种子和packed效率验收。不得把脚本准备、局部合同测试、oracle或初始化探针写为完成训练。

[最新收尾回执](../../experiments/p0-foundation-baselines/20260918-107-awq-continuation/closure.json)记录原生请求execute=true、准备无错误；平台OFF与停止计费未独立核验。汇总数据及上述每项来源SHA详见[结果数据](EVIDENCE_INDEX.json)。
