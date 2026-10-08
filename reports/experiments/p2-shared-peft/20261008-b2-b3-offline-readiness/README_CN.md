# B2/B3 前置工作与代码审查（2026-10-08）

**结论：离线前置工作完成，可以在用户下次开启的实例上从真实10步烟雾开始执行。GPU烟雾、B2/B3训练与闭环结果均未产生。** 本报告是软件准备证据。

## 原阻断与修复

| 原阻断 | 修复与CPU证据 | 仍需实机验证 |
|---|---|---|
| B2不能冻结B1后训练视觉 | 独立训练入口；196个视觉transformer Linear白名单，两个卷积排除；零输出、全部梯度、冻结B1/基座不变 | 真实反传接口、显存和10步 |
| evaluator只能加载单个语言state | 明确语言/视觉双state，分别验证SHA/key/shape后合并，禁止重叠 | OpenVLA装载及配对微测 |
| B3旧20-block loader拒绝32-block | B3专用完整224目标/448矩阵；旧B0/B1不放宽。缺失、非有限值、错shape均拒绝 | 真实Response-SVD及梯度 |
| A4数据未独立采集 | 新A4配置/422目标检查；采集40训练回合的80观测，拒绝A3替代和成功筛选 | A4 capture完整/有限动作 |
| 长评测协议证据不足 | 固定argv/物料SHA、micro审计、各50首片硬gate、全观测SHA/动作trace核验；负收益也能过协议gate | 独立条件未建立，不默认长片 |
| 开机后才写基础代码 | 本轮离线实现、CPU测试、计划生成器、执行卡完成；未连接服务器 | 下次绑定实际实例/收尾工具 |

CPU使用缩小通道的完整目标树；未加载7B checkpoint，不能证明4090可容纳或实际接口通过。四个profile指纹绑定实现文件，以及历史`probe.py`、`low_rank_recovery.py`、`awq_interventions.py`保持不变。

## 本轮复核与验证

实际清单含DINO23块×4、SigLIP26块×4，共**196个视觉Linear**。初版数量检查误写192，CPU测试立即拒绝；已按冻结422清单纠正。真实shape/参数量下次从加载模型导出。

源轨迹划分与仓库镜像字节SHA不同，均明确登记，未声称逐字段相同；实际选用文件上的80帧角色、长度、instruction与样本SHA重新核验并锁定具体版本。75条PEFT留出含历史PTQ暴露，不是全流程盲测；本次未触碰前向/标签。

相关测试：`python -m pytest tests/test_extended_peft.py tests/test_registry_data_contract.py tests/test_recovery_lora.py tests/test_on_policy_capture.py tests/test_e2e_distill_loss.py -q`，**29 passed**；CLI解析与Python编译通过。实际输出及源码SHA在同名[results](../../../../results/experiments/p2-shared-peft/20261008-b2-b3-offline-readiness/README_CN.md)。

检查覆盖冻结/零输出/梯度/重载、B3完整32层与旧loader、角色/reset/query来源、训练评测重叠、烟雾artifact不得冒充完整候选、负结果协议放行、错初态拒绝和不可变计划的trace/first50边界。

## 科学结论及后续

已有B1开发收益284/300，A3=244、BF16=292、A0=294不变。本轮新增闭环样本为0。B2主比较B2−B1；B3主比较B3−A4，B3−B1同时变底座与scope，仅作描述。

80观测训练仍有过拟合和教师能力局限；开发正结果不证明独立泛化。rescue/break提示冲突，不能单独证明可部署路由。Router/P4/P5锁定。

下次：实例/额度/收尾工具→B2烟雾/1000步/micro/first50→分析→额度允许时B3 A4采集/烟雾/1000步/micro/first50。每项一版本、单阶段3小时、整计划墙钟6小时；首片之后先分析/再预注册，无合规项则备份关机。

## POST-RUN DECISION（软件准备）

- RESULT：CPU CONTRACT SUPPORTED；GPU/闭环INCONCLUSIVE（未运行）。
- ALLOWED：CPU缩小模型与数据夹具上的执行契约通过。
- NOT ALLOWED：B2/B3真实烟雾已通过、已获闭环收益/独立复现或INT2部署加速。
- STAGE：留在P2；BRANCH：两项版本冻结，待实机烟雾。
- NEXT QUESTION：当前实例真实10步能否满足冻结、完整装载、有限梯度和重载契约？

执行：[运行卡](../../../../docs/B2_B3_NEXT_BOOT_RUNBOOK_20261008_CN.md)、[B2卡](../../../../docs/pre-run-cards/B2_VISUAL_INCREMENT_V2_20261008_CN.md)、[B3卡](../../../../docs/pre-run-cards/B3_ALL_ELIGIBLE_W2_V2_20261008_CN.md)。代码/小结果/报告进入GitHub；大模型/数据保留既有持久盘与本机归档。本轮没有新服务器归档或关机回执。
