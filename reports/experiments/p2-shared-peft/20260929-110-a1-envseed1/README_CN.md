# A1：110 环境种子 1 评测条件审计与完整闭环结果

**结论：A1 的独立评测条件门禁未通过。** 五配置在同一 450 个 task/reset 上完整运行，数值上 C3 比 C0 多成功 21 回合；但相对上一轮 `env_seed=0`，四个可逐回合对照的配置共 1800 个成败标签全部相同，三个同边界分片的 C3 `policy-queries.jsonl` 甚至逐字节相同。真实评测器调用顺序的无模型复查显示，两种环境 seed 在 10 个任务的 query0 及随后固定 8 步观测都完全相同。该运行应归类为**协议回归重复**，不能算独立条件上的 LoRA 闭环收益复现，不能据此放行 A2 或 Router。

## 问题、物料与执行

唯一拟改变的评测条件是 `env_seed: 0→1`。冻结 exact-12L 混精底座、blocks18–19 的 C3 student-state80 rank8 Recovery-LoRA、C1 旧 demo-only LoRA、14L 静态参照、BF16、模型 seed、官方初态索引 5–49、评测器与 8 步 chunk。A1 没有训练新 adapter，未触碰 `offline_final_holdout`。六个不可变分片按 5–9、10–19、20–29、30–34、35–39、40–49 顺序各跑 C0/C3/C2/C1/BF16；每组总计 450 回合，退出码均为 0，零 episode error，task/reset/初态 SHA 严格配对。30–39 因额度边界预注册拆成两个片，未重复执行。

冻结 SHA256：评测器 `0889196c6a9863fc586f53ecc0831fa46afb3952d576c099e0f6bca765759299`，LoRA 实现 `2fabd32749f52e7b269e3fe66d0e939b2db17a004e1bc4b504e71191de5891f9`，C3 adapter `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`，C1 adapter `ce34247710923b1ff2db9f40c5a0b982cdd228a5238d19b4dd9005e4743c9a47`。三份 AWQ profile 及每阶段实际命令见原始目录的 `CONTRACT_SHA256SUMS.txt` 和 `command.txt`。C3 文件 2,508,930 字节，1,249,280 可训练参数；本轮零训练步。闭环使用 fake-quant，不能据此声称真实 INT2 kernel 加速。GPU 精确活跃秒数尚未从本轮日志独立汇总，记为未测。

## 数值结果与失败模式

| 配置 | 成功/450 | 相对 C0 |
|---|---:|---:|
| BF16 | 438 | +70 |
| C0：exact-12L | 368 | — |
| C1：旧 demo-only LoRA | 366 | −2 |
| C2：14L 静态 W4 参照 | 388 | +20 |
| C3：student-state80 LoRA | 389 | +21 |

同条件内部 C3−C0=+21/450，35 个救回、14 个新增失败；任务分层配对 bootstrap 95% 区间为 +1.78 至 +7.56 个百分点，精确 McNemar `p=0.00380`。五个任务净改善，三个任务净退化。任务 1 为 +14、任务 5 为 +9，两者合计 +23，超过总净增 +21；任务 6 为 −3、任务 7 为 −4、任务 3 为 −2。C3 仍比 BF16 少 49/450。C2−C0=+20，C3/C2 仅差 1 回合；`21/20=105%` 只是这一批开发初态上的描述性比值，不能推断 C3 优于静态 14L。35–39 单片 C3 41/50 低于 C0 42/50，已保留负结果。

这些数值**与上一轮 env_seed=0 的 C0/C1/C2/C3 总数及逐任务结果完全相同**。[`cross-run-outcome-audit.json`](../../../../results/experiments/p2-shared-peft/20260929-110-a1-envseed1/cross-run-outcome-audit.json) 从上一轮原始归档和本轮六个逐回合 JSONL 重新配对 450 个键：450 个 env seed 全改变，450 个初态 SHA 与 model seed 全不变，C0/C1/C2/C3 各自成败变化都是 0/450。C3 的 5–9、10–19、40–49 三个同边界片的原始 `policy-queries.jsonl` SHA256 新旧分别完全相同：`30f90b42111c27254e83c1bd9b219b75d35fe6ba6fd543e5ede364a0844ccac0`、`1049471a60ee9a5de5d9d6cb6af17616aa41b165ed438730a614c913baa7586b`、`d0921d7d3e76075f50319d2ebe865a4fa705ce48db06848851ef021e4dc97b26`。20–29 的文件哈希不同，30–39 因分片边界不同未做文件级等同比较；不能把三个相同哈希推及全部原始轨迹，但逐回合成败全相同已核实。

## A1 失效原因与 Gate

原固定动作预检查只执行 `env.seed → env.reset → set_init_state`，在 reset5 看到 8%–12% 像素差异，于是曾误判新环境流会进入策略。冻结评测器实际执行 `env.seed(environment_seed) → seed_all(model_seed) → env.reset() → set_init_state()`；`seed_all` 随后重置 Python、NumPy、Torch 和 TensorFlow 随机流。补做的 `evaluator-order-probe.json` **完全照真实顺序**，10 个任务在 query0 和固定 8 步后的 agentview、手眼图像、sim qpos、末端位置、夹爪 qpos 哈希均 10/10 相同。结合原始策略轨迹哈希，这一 A1 seed 变更没有建立新的策略可见评测条件。全局重设是最直接的代码层解释；是否存在环境内部其他随机流未被覆盖仍未独立证明。

预注册的成功差、区间和任务数**数值门槛通过**，但前置的独立条件门禁失败，因此研究结论是 `A1_INVALID_CONDITION`，不把它计作第二次独立复现，也不对同一条件重复计算显著性。A2 训练 seed 复现暂停；P4/P5/Router、backbone 改动、静态 W4 搜索继续锁定。

下一步先设计**真正可观测且来源清晰的新评测条件**，例如在 `set_init_state` 之后施加预注册的单一、可复现扰动，或取得与训练/开发不重叠的新初态数组。必须先用与真实评测器同顺序的无模型检查验证 query0/8 步观测和物理状态确实不同，记录哈希、扰动幅度及训练重叠审计，再单独立版本化 PRE-RUN CARD；若不能一次只改变一个主变量，就不运行 GPU。任务 1/5 的收益集中和任务 6/7 的退化仍需后续同观测动作/失败时序诊断，不能据此宣布环境驱动的路由可行。

## 产物与备份

小型配对结果、完整 450 汇总、交叉运行审计和真实顺序预检查在同名 `results/experiments/p2-shared-peft/20260929-110-a1-envseed1/`。服务器原始视频、日志与动作 trace 位于 `/root/autodl-tmp/qvla-repro/eval/a1-110-envseed1-{5-9,10-19,20-29,30-34,35-39,40-49}/`，大文件不入普通 Git。服务器持久盘完整归档为 `/root/autodl-tmp/qvla-repro/backups/a1-110-envseed1-complete-20260929-221049-852d350/`；本机同名副本位于被 Git 忽略的 `backups/experiments/p2-shared-peft/20260929-110-a1-envseed1/`，归档 10 项 SHA256 全部一致。真实顺序预检查 JSON 还保留服务器 `/root/autodl-tmp/qvla-repro/artifacts/a1-110-envseed1-preflight-20260929/evaluator-order-probe.json` 并提交 Git。

GitHub `zsure27/VLA-Quant` 在关机前已核验结果与分析提交 `8d410f308a6a726a552417a1408d8cb3b682749f`；关机回执提交 `764611b5d2c4fede013c4371bf9b0da19f719bd8` 也已核验远端 SHA。`closure.json` 记录 110 的原生 supervisor `837` 关机请求 `execute=true`，SSH 随后断开（退出码 −1），快速 fallback 没有准备错误；本轮低频心跳已设为 PAUSED。平台控制台 OFF 和停止计费未独立核验，不将原生请求回执等同于平台状态。本轮不关闭 Windows 本地电脑。
