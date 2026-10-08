# B2/B3 下次开机执行卡（2026-10-08）

**当前状态：代码已实现、CPU 契约已验证；真实 OpenVLA GPU 烟雾仍待下次用户开机执行。** 本轮没有服务器连接、GPU 训练或新闭环结果。两项独立进行，一次只运行一个计划。

## Research Governor 与已有证据

PRIMARY=P2 shared PEFT recovery；P4/P5/Router 锁定。主问题是静态低成本微调还能恢复多少能力。采用最新模型注册表、20260930-059 结果/决策和 20261005-035 物料审计；历史 seed-only EVAL_REPLICATION 无效，不作为复现证据。

同条件开发 reset20–49：A3=244/300、B0=260/300、B1=284/300、BF16=292/300、A0=294/300。B1−A3 为45 rescue/5 break。B1−B0 的旧任务聚类下界未过确认性门槛；用户授权的 B2 是探索性增量，不能改写旧门禁。两项仍是复用开发初态上的工程比较。

| 项目 | B2 | B3 |
|---|---|---|
| 固定底座 | A3 exact-12L + 冻结 B1 | A4，全422个合格AWQ目标W2；保护模块高精度 |
| 唯一主要干预 | 新视觉rank8 adapter | 新语言32-block rank8 adapter |
| 训练参数 | 双视觉塔196个 transformer Linear；392个矩阵 | 32×7=224个语言Linear；448个矩阵 |
| 不训练 | B1、量化权重、卷积、projector、head、proprio、norm | 视觉塔、量化权重及保护模块 |
| 初始化 | 标准零输出 | peft_train校准80帧的Response-SVD |
| 数据 | 原B1 student-state80，manifest固定SHA | A4新采集40个训练回合×前两次查询；80个观测 |
| 主差值 | B2−B1 | B3−A4 |
| 同条件参照 | A3、BF16、A0 | B1、BF16、A0 |

B2/B3不是同参数预算的胜负比较；均须报告真实参数、序列化字节、训练秒数、显存和配对收益。当前为BF16存储fake quant，不能报告实际INT2压缩/加速。B3的“全W2”只覆盖AWQ合格目标。

## 已补齐的执行链

- `qvla/extended_peft.py`：固定目标白名单、rank8/shape/finite/精确key、训练reset/query与数据来源门禁、动态候选artifact注册。
- `scripts/train_extended_peft.py`：模型加载前审计输入，BF16同观测标签；B2冻结B1只训练视觉；B3在A4上训练全部32语言block。10步与1000步分目录；正式训练从烟雾前的初始化重新开始，不接着烟雾模型累计1010步。
- 烟雾检查真实模型零残差等价、全部应训练参数有限梯度、所有冻结参数/缓冲区前后SHA相同、保存重载后动作完全相同。B3恢复非零SVD初始化前独立验证零分支。正式训练重新核对烟雾代码、数据、初始adapter、教师标签与目标shape。
- evaluator新增明确的视觉state入口及B3专用32-block装载；旧B0/B1 loader仍拒绝额外层。B2/B3规范评测必须带full-training artifact、profile SHA和训练manifest，烟雾checkpoint不能冒充完整候选。
- 新A4数据采集器：只取reset0–3的query0/1；不按成功筛选，不能用A3学生数据替代。
- `prepare_extended_peft_plans.py`：生成两份版本化不可变计划；`run_locked_peft_stage.py`每阶段核对hostname/代码/物料SHA并限制超时，失败终止整个子进程组。
- `audit_extended_peft_eval.py`：检查实际argv、退出码、全部manifest、首观测配对、每个观测文件SHA、8步/7D有限动作及完成数；计算rescue/break/净差、任务聚类区间、McNemar和同条件参照差距。负成功率仍可通过协议门禁。

## 输入与哈希

profile/B1/原student-state80沿用`configs/model_registry_v1.json`的固定SHA。继承SVD清单在035已实测80/80均是peft_train、80条唯一轨迹，源manifest=`ae493d1fe3e32a5e64fb3658fee19b3528fb1c65ecd14c7f92f14ecb48d13742`；下次仍复核实际文件。

轨迹划分存在两个已登记的字节版本：035源文件=`d4903a2e9a1ee6b474f5215644ba68d494a8feb89354b3a2a88bbe13a9d48124`；仓库镜像=`3560425653ef4b297a48a3854aae20348898129b6673db1474943e95897606a2`。本轮只核验镜像字节，未取得源文件重新做完整逐字段比较，**不声称两者逐字段相同**。入口仅允许这两个明确版本，并对实际采用文件逐条核验80帧的轨迹role/instruction/长度/样本SHA，计划锁定实际字节版本；其他SHA拒绝。独立性审计仍需完整来源排除。

75条离线留出为PEFT留出；6条历史PTQ暴露已记录，不是全流程盲测。本轮及后续B2/B3不查询它。训练reset0–3与开发reset20–24分离；已重复使用的开发初态不能变成独立验证。80个观测仍是小样本，1000步重复拟合；扩大scope也可能过拟合，训练loss不能代替闭环收益。

## 下次开机的固定顺序

1. 用户提供当前实例后，SSH核验实际hostname、GPU、Git版本和数据路径；同步并编译核验原生收尾工具。读取当前五小时额度。无可用合规资源时及时保存问题并关当前实例。
2. 使用`configs/experiments/b2_b3_materials.template.json`生成实例专属materials，替换`REPLACE_WITH_CURRENT_CHECKOUT`；逐项根据实际克隆路径定位，不更改内容SHA。若B1归档只在tar中，先按归档SHA恢复，不用其他同名adapter。
3. 在冻结OFT overlay环境下注册两份计划（变量只代表已核验的实际路径）：

   ```bash
   export PYTHONPATH="$REPO/diagnostics:$REPO:$OFT_ROOT:$LIBERO_ROOT"
   "$PY" "$REPO/scripts/prepare_extended_peft_plans.py" \
     --materials "$MATERIALS" --session "$SESSION" --output "$PLANS" \
     --python "$PY" --expected-hostname "$HOSTNAME_VERIFIED"
   ```

   `SESSION`须为持久盘`backups/experiments/p2-shared-peft/YYYYMMDD-INSTANCE-b2-b3/`；`PLANS`写入持久盘control目录。注册器只生成计划，不启动GPU。
4. 用`nohup "$PY" "$REPO/scripts/vla_stage_supervisor.py" "$PLANS/B2-plan.json" > "$PLANS/B2-supervisor.log" 2>&1 &`后台独立启动B2，启用仅本轮活跃期间约15分钟心跳。B3同理，不能同时启动两者。
5. B2顺序为烟雾10→正式1000→五配置各10回合微测→协议审计→五配置各50回合首片→硬审计→`AWAITING_GATE_REVIEW`。微测与首片重叠，统计只报告首片50，绝不相加成60。
6. 同一心跳内分析B2、保存原始数据/成本与POST-RUN DECISION；额度允许且没有影响B3的协议错误时启动已注册B3。B2收益方向不决定B3是否运行，二者问题独立；公共协议错误必须先修复。
7. B3先跑A4训练reset0–3的40回合capture→冻结80观测→烟雾10→正式1000→五配置各10回合微测→审计→各50回合首片→硬审计。SVD spool预留至少24GiB磁盘；若不足不删唯一数据、不以其他初始化静默代替。
8. 现有计划**不含remaining250**。两张首片之后必须先分析；声称独立性需真实评测调用顺序下观测条件验证和完整重叠排除。新开发长片也须另行预注册、锁定样本量/条件；不能按成功率方向挑选。

不需要用户实时回复即可在合规计划内自动接续。GPU空闲、有待执行项时恢复同一计划，不重复完成项；无合规下一项或问题需审阅时清晰报告并备份关机，禁止付费空等。

## 收尾与输出

只用五小时实时额度：约15%停止扩展，归档持久盘/本机/同名reports-results分析并推送固定zsure27/VLA-Quant、核验远端SHA；约10%停止新实验，确保进程终止/GPU空闲后原生关闭当前实例，保留至少3%。暂停本轮心跳。当前无Windows关机授权；以后只按新一轮明确授权执行。

报告必须含逐回合rescue与break两数、任务分层区间、负结果、失败轨迹、成本、可允许/不允许因果结论和下一决策。大量rescue/break仅提示冲突；只有动作前环境上下文能预测且超过task_ID控制，才进一步讨论Router。

本地CPU PASS不能替代GPU烟雾；GPU OOM、实际接口/shape或overlay梯度问题仍可能在下一次10步中暴露，失败时保留产物并按约定及时收尾。

每候选最多一版本、计划累计墙钟上限6小时（包括阶段与间隙）、单阶段3小时；超时杀子进程组并停止后续阶段，不递归扩大搜索。训练每100步保存安全adapter；该checkpoint不等于正式artifact。意外中断须审计部分产物后恢复或收尾，不能删除输出目录直接重复。
