# 后层平滑因果消融

服务器已完成局部及选定完整 Llama block 的 FP32/BF16 对照。单样本结果支持有限精度扰动，但不证明全部模型基线正确。

本入口复用原始教师和校准文件，不修改官方适配器。新增诊断脚本哈希记录在 manifest 中。

运行 `bash scripts/run_smoothing_ablation.sh`。默认复用 `controls-20260911-234754-1133`，需要时以 BASE 指定。

先验证 repeat；然后每个新进程运行一个 W16A16 配置。所有四组均使用教师 projector 输出，视觉平滑仍执行但其 projector 输出被替换。对照变量只涉及 LLM 第 23–31 层。

| 配置 | 保留平滑组数 | 取消内容 |
|---|---:|---|
| all | 162 | 无 |
| no-late-attention | 153 | 后层 Norm 与 QKV 整组 |
| no-late-mlp | 153 | 后层 Norm 与 gate/up 整组 |
| no-late-both | 144 | 上述两类 |

不是关闭 attention，也不改变 mask、温度或后端。不允许这些范围选项用于低比特量化。逐步夹爪记录包含原始值、归一化值及相对现有 rollout 决策规则的带符号距离。

完成不等于 PASS；不会产生控制门槛标记。已有失败结果继续保留。正式运行器现在重新读取真实控制指标，并只在成功后将 pending 文件改名为 pass，避免空文件误放行。

本更新的本地测试只覆盖选择逻辑和语法；完整 GPU 对照由服务器执行。

## 全语言关闭负对照

`--smoothing-selection no-language --oracle-projector` 取消全部 64 个语言平滑组，保留 98 个视觉平滑组，但视觉路径的 projector 输出由教师缓存替换。要求 W16A16；不生成正式 controls.pass.json。预期动作及 LLM 输入/输出恢复教师结果，视觉内部特征仍可不同。失败时继续检查干预流程，不能据此修改阈值。
