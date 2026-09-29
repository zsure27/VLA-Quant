# A1 PRE-RUN CARD：冻结 C3 的独立评测条件复现

**状态：HOLD；本卡尚未放行任何 GPU 回合。** 截至 2026-09-29，本地代码与 GitHub `zsure27/VLA-Quant` 的 `main` 均为 `9e33d4460a6458db555cef5cf3ced88a460765e9`，本地工作树在写本卡前干净。上一轮 107 的原生关机请求已有回执，但平台 OFF 未独立核验；本轮没有新的在线实例或运行产物。此卡遵循上传的 Research Governor A1，并以原始产物而非历史摘要为准。

| 字段 | 预注册内容 |
| --- | --- |
| EXPERIMENT_ID | `a1-c3-eval-envseed1-20260929`（候选 ID，未运行） |
| PRIMARY_STAGE | A1 / shared PEFT 稳定性 |
| SCIENTIFIC_QUESTION | 固定现有 C3 adapter 后，相对 C0 的闭环正收益能否在独立评测条件上复现？ |
| HYPOTHESIS | C3−C0 仍为正，且净收益来自至少两个任务；task1/5 改善及 task6/7 退化均须单独呈现。 |
| BACKBONE_VERSION | `awq-w2a16-12l-mixed-spatial-v1`；12L 只在语言 blocks8–15、20–23 为 W4，其余语言 W2 G64；DINO W2 G64、SigLIP W2 G128。C2 只将 blocks18–19 改为 W4。 |
| ADAPTER_VERSION | C3 student-state80 rank8 Recovery-LoRA，仅 blocks18–19；固定 `candidate-adapter.pt` SHA256 `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`，实测 2,508,930 bytes，1,249,280 可训练参数。C1 使用历史 demo-only adapter，其上轮契约 SHA256 为 `ce34247710923b1ff2db9f40c5a0b982cdd228a5238d19b4dd9005e4743c9a47`，待复算原件。 |
| DATA_UNIT | 主统计单位为 episode；策略每次 query 输出 8×7 动作 chunk，完整执行 8 步后再查询。student-state80 是 80 个 query 观测，不是 80 条轨迹。 |
| TRAIN_SPLIT | 本轮零训练。C3 既有训练采样为 10 tasks×reset0–3×query0/1，同观测 BF16 重标；既有 Response-SVD 校准仍来自 peft_train。 |
| EVAL_SPLIT | 拟保留 10 tasks×官方 reset5–49=450 episodes/配置，作为开发复现；`offline_final_holdout` 继续封存。只有确证新环境种子带来真实新评测条件时才可运行。 |
| ONE_CHANGED_VARIABLE | **仅**把冻结评测器的 `--env-seed 0` 改为 `--env-seed 1`；`--seed 0`、`--seed-protocol paired` 和 reset 索引不变。五种策略共享同一新种子、task、初态与执行协议。不得同时切换新 reset 文件、扰动、评测器或模型种子。 |
| FIXED_VARIABLES | 模型 checkpoint、12L/14L 三份 AWQ profile、clip/group 配方、C1/C3 adapter、评测器与 LoRA 注入代码、LIBERO/依赖、预处理、动作后处理、8 步 chunk、任务顺序、每任务回合数和成功定义。 |
| CONTROL | 同协议 BF16、C0 exact12L、C1 old demo-only LoRA、C2 static14L、C3 fixed student-state LoRA；所有模型在同一新评测条件下重跑，不能混用旧 seed 的分数。 |
| PRIMARY_METRIC | 配对 episode 成功差 `S(C3)−S(C0)`；同时报告各配置绝对成功数。 |
| SECONDARY_METRICS | 按 task 分层的 episode 配对 bootstrap 95% CI、精确 McNemar、rescue/new failure、逐任务净差、task1/5 改善、task6/7 退化、BF16 与 C3 差距、C2−C0 headroom；仅当 `S(C2)>S(C0)` 时计算 `RR_14=(S(C3)-S(C0))/(S(C2)-S(C0))`。另报异常回合、哈希/配对一致性和 GPU 时间。 |
| EXPECTED_DECISION_IF_POSITIVE | 若总净差为正、配对 CI 下界大于 0，且至少两个任务净改善，保留 C3 为可复现 shared PEFT 候选，随后另立 A2 训练种子复现；Router 仍锁定。 |
| EXPECTED_DECISION_IF_NEGATIVE | 若净差消失、反转，或任务退化抵消收益，暂停 Routing，先研究共享 PEFT 的训练分布与稳定性。正差但区间跨 0 记 INCONCLUSIVE，不用它放行 Router。 |
| STOP_RULE | 任一冻结文件 SHA、配对 manifest、动作语义或新条件独立性核验失败，立即停止；不得以更换多个 seed/reset/扰动“补救”。阶段失败先保留原始产物，不重复计数。 |
| GPU_BUDGET | **当前批准值 0。** 若将来放行，最多 5 配置×450 回合=2,250 回合；以 C3 上轮 450 回合约 78 分钟粗估，五配置约 6.5 小时加装载/备份，须按实时五小时额度切成可完成的固定分片并遵守 15%/10% 收尾阈值。该时长不是测量过的五配置总耗时。 |
| OUTPUT_PATH | 计划使用同名 `results/experiments/p2-shared-peft/20260929-a1-evaluation-replication/`；当前仅存预注册契约，不存在回合结果。 |

## 运行前证据与哈希

上一轮完整本机归档位于 `backups/experiments/p2-shared-peft/20260929-107-p25-student-state80-distill/p25studentstate80-20260929-040513-852d350/`。本机已重新计算 adapter、活动代码包、原始结果包和 Git bundle 的 SHA256，均与归档 `SHA256SUMS.txt` 一致。归档内活动代码逐文件重新计算如下，均与每片的 `CONTRACT_SHA256SUMS.txt` 一致：

| 文件/角色 | 上轮 SHA256 | 当前可验证程度 |
| --- | --- | --- |
| C3 adapter | `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a` | 本机原件复算通过；待核验新实例加载文件 |
| C1 demo-only adapter | `ce34247710923b1ff2db9f40c5a0b982cdd228a5238d19b4dd9005e4743c9a47` | 旧 500 回合契约记录；待复算持久盘原件 |
| 12L backbone 配置 JSON | `a30d68788bc2aa101ffa4f46d76def7d93b4402fd293db704a305cf254d2d35c` | 当前仓库复算通过；实际精度还须检查运行参数与 profile |
| 量化评测入口 `run_eval_official_quant.py` | `0889196c6a9863fc586f53ecc0831fa46afb3952d576c099e0f6bca765759299` | 归档与当前仓库复算通过；待核验新实例 |
| LoRA 注入 `recovery_lora.py` | `2fabd32749f52e7b269e3fe66d0e939b2db17a004e1bc4b504e71191de5891f9` | 归档与当前仓库复算通过；待核验新实例 |
| OpenVLA-OFT LIBERO evaluator | `fe37e8097c286e1946e5194a1f817a2b2362d167a6a5a49443c5aa20314d8873` | 归档成员复算通过；待核验新实例 |
| 12L/C2 共用基础 W2 profile `w2.pt` | `4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc` | 上轮契约记录；原二进制未在本机归档，待持久盘复算 |
| W2 G64 profile `w2-g64.pt` | `947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8` | 上轮契约记录；待持久盘复算 |
| 14L W4 profile `w4.pt` | `ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4` | 上轮契约记录；待持久盘复算 |

两种 mixed 配置调用同三份 profile，仅 W4 block 列表不同：C0/C1/C3 为 `8–15,20–23`；C2 为 `8–15,18–23`。上一轮评测器 `--seed 0 --env-seed 0 --seed-protocol paired`，官方 reset5–49，五片均 exit0。模型 checkpoint 目录为 `/root/autodl-tmp/qvla-repro/models/openvla-7b-oft-finetuned-libero-spatial`；本归档没有其完整文件哈希，必须在新实例确认 checkpoint 文件清单与哈希。旧 C1 adapter 的当前字节哈希也须从原始归档/持久盘复算。上述任何缺口不得用路径相同替代哈希核验。

## 当前 HOLD 的实质原因

冻结 evaluator 的 `load_initial_states` 直接读取官方固定 50 个状态；`--env-seed` 仅通过 `env.seed` 改变环境随机流，随后仍以同一索引调用 `env.set_init_state(initial_state)`。所以 `--env-seed 1` **不会生成新的初态数组**。现有源码与归档不能证明它会让首次策略观测或后续仿真轨迹发生有意义的变化；如果两次 rollout 等价，它就不是独立条件复现。不得把不同 env seed 数字本身写成新 reset 证据。

放行 A1 前先做不训练、不使用 GPU 的预检查：在当前已确认实例上复算全部冻结文件 SHA；对冻结 reset 的同 task/索引，只更换 env seed，用 CPU 仿真比较初始观测，并在同一固定动作序列下比较接下来的环境观测/状态，保存两条件 manifest 与哈希。只有出现可复核的非数值噪声差异，才把它视作新评测条件；否则此 A1 方案终止，改为**另行预注册**只有 reset/扰动协议变化、模型种子和所有模型仍冻结的单变量实验。新方案先生成合法独立 reset manifest、核对与训练/开发不重叠和来源许可，不能触碰 `offline_final_holdout` 或把旧 0–49 重新命名。新的方案须重写本卡，不在本卡中同时改 env seed 和 reset。

此前 C3 在开发 reset5–49 的 389/450 对 C0 的 368/450（35 rescue、14 new failure）是提出 A1 的依据，不是 A1 的结果。历史 411/412 与 430/431 漂移尚未归并为一个跨 session canonical 值；本卡只把上轮**同源配对** 450 结果作为参考，不混用历史 500。未完成上述条件之前不启动 GPU，不训练 adapter，不实现 Router，不改变 backbone。
