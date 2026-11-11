# 2026-11-12 061克隆接续收尾覆盖

用户要求暂停并关机；GPU已停，061已取得北京时间11月12日02:12:17原生execute=true回执，SSH随后关闭；平台OFF/计费未独立核验。053→061科学条件逐SHA复核不变，新增61完整和2中断，累计429唯一完整轨迹。初态15共有157个量化配对；BF16 31/32、AWQ29/32、GPTQ30/31、Omni28/30、SQ8 31/32、SQ4 0/32，覆盖不齐，不作公平方法排名。剩51个Long槽位，下一次开机新授权后仅补缺项，不根据旧RUNNING自动启动。

[本轮综合分析和五类实测图](rounds/20261112-061-user-close-v1/README_CN.md)；两个新增原始增量包私有下载SHA及本机1340成员均过。历史1.05GB父包云端缺口、大参数服务器保留边界不变。Windows保持开启。以下均为历史阶段，最新结果以本段和最终回执为准。

# 当前综合结果（2026-11-11）

核心仍是量化前后轨迹漂移现象诊断，不训练、不改进量化方法、不复现DyQ-VLA代码。以[统一评测标准v7](UNIFIED_EVALUATION_STANDARD_V7_CN.md)为准：用户列出的12个量化配置，BF16单列；按实际模型适配门禁推进，未实现和架构不适用分开，不将旧v5配置数当成当前已完成数。

- OpenVLA-OFT：BF16 37/40，AWQ W4A16 39/40，SQ W4A8 37/40，SQ W4A4 0/20。A4覆盖缺20，不能跨不同覆盖直接排名；均仅初态5，未完成五初态。[模型报告](model_reports/openvla_oft.md)。
- π0.5：原生JAX参考39/40；JAX→Torch等价门禁仍失败。独立同Torch后端四suite task0/初态5，BF16 4/4，SQ W4A8 3/4（Spatial新增失败）。只有4对pilot，不与JAX相减；SQ4仅产物门禁，GPTQ仅CPU准备。[最终恢复报告](rounds/20261111-084-final-v1/README_CN.md)。
- GR00T-N1.5：旧Spatial task1成功轨迹单列；11月11日新增Spatial task0/初态5 BF16成功79步，重复/重载与reset0校准完成。Object因旧入口revision常量笔误在模型加载前失败，其他suite及量化未运行。当前没有GR00T量化漂移结论。[本轮结果/备份缺口](rounds/20261111-008-groot-reference-sq-v1/README_CN.md)。

008本轮原始增量包约37.6MB已在服务器逐成员自验，自动收尾窗口后SSH不可达，私有回传、原生execute=true回执与平台OFF均未确认。独立v2已修复revision并加入全四suite预检、父原始包云端恢复门禁和“初始状态已经失败即立即返回”的事件监听测试；下次用户开启后先取包再续，不重跑已完成Spatial。本地Windows保持开启。

以下v5/v4/v3内容保留为历史方案，不再代表当前运行或完成状态。

# 当前任务摘要（v5）

当前只建设独立量化漂移诊断，不复现DyQ代码、不做OPD训练。三模型 × LIBERO四suite × 10任务 × 5初态，每模型每条件200回合；11静态量化配置＋BF16共12条件，QVLA与DyQ动态行排除，完整上限7200自然回合（公共BF16复用）。尚无服务器、新实测或全部方法兼容证据。

- [v5完整设计与调研](DESIGN_V5_CN.md)
- [来源审计](research-v5.json)
- [计划case清单](planned-v5/planned-cases.csv)与[CPU收据](runs/prepare-cpu-v5.json)

五初态前2发现/后3复查；多卡4090只做独立worker分片建议，逐GPU资源/占用锁、真实方法适配和本轮授权未通过则不可派发。旧图仍为合成示意，不能填入新的200轨迹分母。

## v4历史摘要

最新增加OpenVLA-OFT（本轮配置h=8）：[v4设计](DESIGN_V4_CN.md)、[OFT独立六组预览](preview-v4/oft/index.html)、[三模型总览](preview-v4/index.html)。仍无服务器/新实测；原生方法适配及实际GPU门禁未齐。首批预算由64调整为96自然回合，增量32，不默认长评测。新增3项CPU合同测试通过，v3源码与图保持原字节。

## v3历史摘要

无服务器、无GPU、无新实测结论。主范围为π0.5/GR00T-N1.5 × LIBERO四suite × BF16/ActQuant-W4A16/SmoothQuant-W4A8/QuantVLA-W4A8。先64自然回合；可选短扰动探针独立注册，不跑默认450或方法全网格。π/GR原生采集与量化适配尚未实现，不能复用OFT绑定冒称兼容。

- [完整v3设计](DESIGN_V3_CN.md)：同精度扰动分支、误差t→动作t+1、原生5/1步、风险集、预算与限制。
- [v3六组图](preview-v3/index.html)：联合分布、中心、离散度、响应、共性矩阵和风险集。
- [v2七组图和空间轨迹播放器](preview-v2/index.html)：空间偏移、W/A分解、动作块及事件对齐。
- [v1六组图](preview/index.html)：历史基础视图。
- [v3 CPU收据](runs/prepare-cpu-v3.json)：新增7项CPU检查；所有图为合成示意。

完整同任务日志在[任务卡](../../20261106_1504_四套件多任务轨迹偏移诊断.md)。v1/v2历史方案不再作为主起跑指令，不覆盖冻结源码或GPU门禁。
