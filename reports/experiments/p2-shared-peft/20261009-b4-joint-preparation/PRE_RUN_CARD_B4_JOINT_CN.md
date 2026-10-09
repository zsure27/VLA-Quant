> **300回合覆盖：**用户已将下一轮正式新鲜配对评测改为B3/B4各300（reset20–49）。训练定义沿用本卡，评测/成本/对照口径以[新PRE-RUN CARD](../20261009-b4-300-preparation/PRE_RUN_CARD_B4_PAIR_300_CN.md)为准；本卡200回合与同期五配置安排为旧版本。

> **2026-10-09后续修订：**本卡的训练定义保留，五配置重新评测安排由[数据范围/效率新协议](../../../../docs/DATA_SCOPE_AND_EFFICIENT_EVALUATION_20261009_CN.md)覆盖。实际训练/测试共享全部十任务，只作同任务开发证据。新正式GPU回合仅B3/B4各200；全内容数据范围审计尚待原始归档，未通过前不得启动GPU。旧CPU7项仍为历史准备证据，新修订CPU13项与源文件SHA见[新审计](../20261009-data-scope-audit/README_CN.md)。

# PRE-RUN CARD：B4视觉＋语言联合Recovery-LoRA

状态：预注册/CPU契约，未运行GPU。用户2026-10-09明确授权；先完成上轮046结果校验和完整双份归档，再执行B4。

- **PRIMARY STAGE**：P2 shared static PEFT recovery。
- **CLASSIFICATION**：MODEL-SELECTION / DEVELOPMENT；合同烟雾另记CONTRACT。
- **QUESTION**：A4上扩大为视觉＋语言联合可训练范围，是否比同预算训练步数的B3得到净闭环增量？
- **HYPOTHESIS**：语言恢复后剩余的部分失效可能需要视觉与语言共同修正；增加视觉可训练因子可能带来rescue，也可能引入break或过拟合。
- **BACKBONE HASH**：`configs/backbones/awq_w2a16_all_eligible_spatial_v1.json`与三份profile固定SHA见同名results的preparation-audit.json；启动时核验实际checkpoint完整identity和effective 422目标。当前实例/环境未绑定，不填写虚假实测值。
- **METHOD**：A4＋224语言Linear＋196视觉transformer Linear，rank8逐层独立因子，联合最终8×7动作Smooth-L1蒸馏；26,710,528可训练参数。
- **DATA SPLIT**：精确复用B3训练reset0–3的40回合/80观测、冻结BF16同观测标签；不使用B1学生状态。语言初始化复用B3训练前Response-SVD，校准来源保持80条peft_train轨迹。评测官方开发reset20–39，非独立A1、非盲测；holdout锁定。
- **ONE CHANGED VARIABLE / VARIABLE**：可训练目标由语言224扩为语言＋视觉420个Linear。容量和计算随范围共同增加，结论仅为工程总效果，不能声称同参数比较或纯视觉归因。
- **CONTROL**：底座、profile、clip/group、语言初始化、80输入、教师标签、1000步、batch1、AdamW、lr/wd、loss、seed/order和评测语义固定。比较冻结B3；A4、BF16、A0均同期评测。不能从训练后的B3继续1000步后归因范围。
- **PRIMARY METRIC**：严格配对B4−B3成功差，rescue与break、任务聚类95%区间；完整200/配置。
- **SECONDARY METRICS**：B4−A4总恢复、与BF16/A0差距和四格、任务分层、失败模式、参数/文件字节/GPU时间/显存/推理成本。
- **DECISION / EXPECTED DECISION**：净差>0且任务聚类区间下界>0则支持当前开发分布增量，评估其成本后保留B4候选；只有方向而无充分区间支持则结论INCONCLUSIVE。独立条件验证另立计划，不放行Router。
- **STOP_DECISION**：无净增量或增加破坏则保留B3简单默认，关闭本分支，转向已保存失败的P2.5机制诊断；不自动扩rank或专家。
- **MAINLINE_RELEVANCE**：确定简单共享静态修复是否足以逼近W4/BF16，以及额外视觉参数是否值得，之后才决定互补/路由研究必要性。
- **STOP CONDITION**：旧归档/物料/哈希/数据角色不通过，OOM、非有限梯度/动作、冻结或重载不等价、缺trace/配对失败、资源/6小时预算/实时额度门限。续跑只按协议，不按成功方向。
- **EVALUATION**：五配置B3/B4/A4/BF16/A0，固定seed0/env_seed1/paired、完整8步chunk；reset20微测10/配置不计正式；reset20–24的50＋25–29的50＋30–39的100=200/配置，共1000正式回合。
- **OUTPUT DIRECTORY**：开机后绑定`backups/experiments/p2-shared-peft/YYYYMMDD-INSTANCE-b4-joint-v1/`；control独立不可变计划；Git报告/小结果同名reports/results。准备材料会话为20261009-b4-joint-preparation。
- **ESTIMATED COST**：新增6.72M训练参数；因子裸BF16约53.42MB。训练分钟级至数十分钟、评测小时级，仅为规划；10步后按实际GPU时间/显存更新，三计划累计墙钟最多6小时。旧B3训练峰值约18GiB，B4视觉反传存在24GiB OOM风险，GPU烟雾必须先测。
- **CURRENT LOCKED STAGES**：P4、P5、Router、offline_final_holdout、scale/clip/group搜索。

实施顺序与后续门禁见[下次开机runbook](../../../../docs/B4_NEXT_BOOT_RUNBOOK_20261009_CN.md)。所有未测指标留空，不把计划或CPU结果当闭环证据。
