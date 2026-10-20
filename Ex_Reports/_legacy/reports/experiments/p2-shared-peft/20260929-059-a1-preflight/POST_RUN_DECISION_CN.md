# 059：A1 运行前门禁与停止决定

状态：**A1_HOLD / 没有闭环回合 / 没有 GPU 策略实验**。本文件记录 2026-09-29 在新机 059 上的运行前检查；它不是 A1 成功率报告。预注册问题与统计判据见 [A1 PRE-RUN CARD](../../../../docs/pre-run-cards/A1_EVALUATION_REPLICATION_20260929_CN.md)。

## 已完成

- 新端口 `21271` 的独立 host-key pin 与 SSH hostname `autodl-container-5e9a44960d-2363ecb7` 核验通过；未把旧 107 主机身份用于 059 关机。
- 启动实验前已同步快速备份/原生关机工具，并通过远端 SHA256 与 Python 编译检查。
- 059 实际 C3 adapter SHA256=`67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`，C1 adapter SHA256=`ce34247710923b1ff2db9f40c5a0b982cdd228a5238d19b4dd9005e4743c9a47`；基础 W2=`4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc`、语言 G64=`947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8`、W4=`ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4`，均与历史契约一致。
- 量化评测器 `0889196c6a9863fc586f53ecc0831fa46afb3952d576c099e0f6bca765759299`、LoRA 注入 `2fabd32749f52e7b269e3fe66d0e939b2db17a004e1bc4b504e71191de5891f9`、OFT LIBERO evaluator `fe37e8097c286e1946e5194a1f817a2b2362d167a6a5a49443c5aa20314d8873` 与原归档一致。模型 checkpoint 四块 safetensors、action head、proprio、adapter 和配置文件逐项与 2026-09-27 原始 `checkpoint_identity.files` manifest 匹配；该 manifest 记录的汇总 SHA 为 `b909ec7ebc9d41890a7b956118763af4a78e408ba7a39106d2be4b3109918340`。
- 完成检查时 4090 没有 compute 进程。没有训练新 adapter，没有启动 Router，没有修改 12L/14L backbone 或正式评测器。

## 未通过的科学门禁

冻结评测器对同一官方 reset 索引加载固定初态；`env_seed` 不改变初态数组。A1 只有在环境种子改变实际可观察状态/轨迹时才有意义。为此准备了固定动作、无策略权重的 probe（SHA256 `80acba4d985dcfbdd3b370e11b6e61f1d8c8bf199bc7e011ac149676c07f1b7b`）。首次执行将 `CUDA_VISIBLE_DEVICES` 设为空，EGL 渲染器在初始化时报 `ValueError: invalid literal for int() with base 10: ''`；没有产生有效样本或 A1 分数。失败日志保留在服务器持久盘 `artifacts/a1-preflight-059-20260929/`。

拟使用 GPU 0 完成同一无模型仿真 probe 的重试被**自动审批审查拒绝**。审查理由：CPU-only probe 失败后分配 GPU 0，与“独立评测条件未建立前停止 GPU 工作”的明确门禁冲突；并要求不得以间接方式绕过。故没有重试，也没有派发 C0/C1/C2/C3/BF16 闭环。此处是方法学/执行门禁受阻，**不能将 A1 记为阴性或 C3 收益未复现**。

## 决策与后续

A1 继续 HOLD；P4/P5、Router、A2 和任何新 adapter 训练仍锁定。下一次只有在能以明确允许的方式证明 env seed 产生真实新条件，或另行预注册合法的独立 reset/扰动作为**唯一**评测变量后，才运行五配置配对闭环。不能同时改 seed、reset、评测器、backbone 或 adapter，也不能把旧 0–49 改名为盲测。059 当前无合规 GPU 阶段可运行，因此按“不让服务器空跑”约定归档并关闭该实例。GPU 闭环成本=0；CPU 检查及同步耗时以系统记录为准，不宣称为模型推理时间。

原始 probe 脚本、失败日志、决策和 SHA 清单保存在服务器持久盘；本机副本与 GitHub 状态在同名 `results/experiments/p2-shared-peft/20260929-059-a1-preflight/` 记录。若 GitHub 无法以固定 `zsure27` 身份非交互推送，保留服务器和本机副本并明确记为待同步。

## 收尾回执

服务器持久盘归档为 `/root/autodl-tmp/qvla-repro/backups/a1preflight059-20260929-122953-852d350/`，本机同名副本在 `backups/experiments/p2-shared-peft/20260929-059-a1-preflight/`；两端 `SHA256SUMS.txt` 均通过，10 个归档文件已核验，原始诊断包 SHA256=`0c11766216427209405fadb35d3cf596fbb719dc62433bd5752f9af30eca684b`。关机前 GitHub `zsure27/VLA-Quant` 远端 main 已核验为 `c194b545df005a90fddfb01053957ca42add60d8`。

2026-09-29 04:34:06 UTC 对 059 的 supervisor PID 837 发出 `execute=true` 原生关机请求，工具回执中的原生脚本 SHA256 为 `0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`；随后 SSH 由远端关闭。**平台控制台 OFF 与停止计费未独立核验**。完整回执见同名结果目录的 `closure.json`；本段和回执的补充 Git 提交将在关机后单独核验。未执行 Windows 本机关机，本轮没有该一次性授权。
