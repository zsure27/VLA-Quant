# 四suite W/A诊断配置 v2

全部为阻断模板，未启动或验证任何实际模型。沿用原研究任务身份，新增阶段和round_id，禁止覆写v1计划/输出。

- `bf16`、`awq-w4a16`、`awq-w2a16`：W阶段，120条配对轨迹上限。
- `sq-w4a16`、`sq-w4a8`、`sq-w4a4`：A阶段，仅SQ只平滑等价与同权重/坐标门禁通过后允许，另120条上限。
- 6份`identity.example.json`中的SHA/coordinate/cohort只是占位，必须填实际内容身份；配置中的门禁SHA也必须替换。
- SQ模板使用`smoothquant_diagnostic_v2`显式alias，启动plan/run时都要加`--plugin qvla.trajectory_infra_v2`。alias只说明接口，不是兼容性或精度证据。
- SQ-W16A16/C0与W16A8/A4为离线同输入控制，不默认生成闭环计划。当前四分支重放GPU调度器需在实际服务器上独立准备和验证。
- `study.json`列出8类候选方法与准确实现状态，GPTQ/OmniQuant/DuQuant等尚无OFT绑定，不允许把名称改成现有AWQ后端来执行。

全部准备情况及图形解释见`Ex_Reports/_tasks/multisuite-trajectory-deviation-v1/DESIGN_V2_CN.md`。没有服务器，不创建实验心跳、不启GPU。
