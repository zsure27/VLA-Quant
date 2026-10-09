# 2026-10-09 B4联合微调与下次开机顺序覆盖

用户授权B4=A4全合格W2底座＋224语言和196视觉transformer Linear的rank8 LoRA联合训练；PRIMARY=P2，主问题B4−B3闭环增量。执行docs/B4_NEXT_BOOT_RUNBOOK_20261009_CN.md及独立PRE-RUN CARD。精确复用B3的80观测、BF16教师cache、训练前Response-SVD语言初始化；视觉零输出，联合1000步。不得从已训练B3续训、重采样或改量化配方；新增范围同时增加容量/计算，只解释工程总效果。独立注册configs/model_registry_b4_v1.json，保持旧B3哈希绑定源文件字节。

下次用户自行开机后，先补齐046完整本机归档，核验服务器13项/18494项与本机全部文件、完善旧结果/收尾报告，推送固定zsure27/VLA-Quant并核验远端SHA，取得PASS_PRIOR_CLOSURE；之后才注册B4、真实GPU10步烟雾、1000步训练和配对评测。B3/B4/A4/BF16/A0各200，官方开发reset20–39，分50+50+100，微测不相加，不称独立复现。续跑门禁只依据协议与资源，不按成绩方向选择。P4/P5/Router/holdout继续锁定。

当前7项CPU契约通过，真实GPU烟雾待开机；046由用户手动关闭，本机完整归档未通过。启动前≥12GiB持久盘空闲并留6GiB归档余量；视觉反传显存需实测。按实际hostname/端口同步收尾工具；活跃期约15分钟低频读status/runner/GPU，无变化静默，禁止频繁唤醒等待。约15%实时五小时额度备份、约10%关当前实例、留3%；无合规项或不可修复问题需审阅则报告后备份关机，不空等。旧Windows关机许可不继承。

# 2026-10-08 B2/B3 前置执行覆盖

B2/B3当前实现与CPU契约见`docs/B2_B3_NEXT_BOOT_RUNBOOK_20261008_CN.md`和两张20261008 V2 PRE-RUN CARD；GPU烟雾尚待实机，不把CPU PASS当GPU PASS。B2=A3+冻结B1+196个视觉transformer Linear增量rank8；B3=A4+新32-block语言rank8，必须单独采集A4 student-state80。使用版本化入口与不可变计划，10步烟雾通过才正式1000步；正式训练重新从烟雾前初始化开始。规范装载必须有artifact/profile/训练manifest SHA，旧B0/B1门禁不放宽。微测与first50重叠不得叠加样本；首片之后硬停止并分析，不默认启动remaining250。只在用户自行开机后绑定当前hostname/端口，先同步收尾工具，再启用本轮约15分钟低频心跳；无合规项立即备份关机。Router/P4/P5及holdout继续锁定。历史本机关机许可不沿用。

# 2026-10-05 阻断时避免付费实例空跑

若实验出现无法在当前轮解决的技术问题，或继续运行必须等待用户审阅或选择，立即向用户清晰输出问题、已核验事实、可行方案及下一步门禁；同时按收尾协议保存原始产物、分析、服务器持久盘与本机副本，推送并核验固定 `zsure27/VLA-Quant` 的远端 SHA，关闭当前已核验实例并暂停本轮心跳。不得为等待确认而让 GPU 或付费实例长时间空闲。任何关键备份或远端关机失败时报告失败与现存副本，不关闭本地电脑；本地电脑仅在本轮用户明确授权且全部前置核验成功后关闭。此规则适用于以后所有实验，不能用未完成的审阅门禁放行 GPU 长实验。

# 2026-10-05 模型命名与严格数据审计覆盖

BF16保持不变；A0=W4、A1=16L、A2=14L、A3=12L/旧C0、A4=全合格AWQ W2（语言/DINO G64、SigLIP G128）；B0=旧C3、B1=旧LW、B2=A3+冻结B1+视觉增量、B3=A4+新语言32-block LoRA。后续按 `configs/model_registry_v1.json` 和 `docs/MODEL_NAMING_AND_DATA_AUDIT_20261005_CN.md`。旧实验阶段A1/A2写作EVAL_REPLICATION/TRAIN_ORDER_REPLICATION，原日志不改。

用户已授权语言W2全范围扩展及预注册B2/B3探索，覆盖此前“所有PEFT只能blocks18–19”的当前范围描述；B0仍是18–19，B1是20个语言W2 blocks。B2/B3尚未通过实现与烟雾门禁，禁止直接长跑；Router/P4/P5仍锁定。新规范评测必须有model-id、注册profile/adapter SHA和训练manifest，拒绝训练reset与评测重叠。恢复训练必须在模型加载前通过role、轨迹划分和策略可见观测内容检查，路径/文件名不同不构成不重叠证据。

历史W4 AWQ32校准与后续切分交叉核对为18条peft_train、8条router_dev、6条offline_final_holdout。因此75条仅是PEFT留出，不是全PTQ/基础checkpoint流程留出；其他profile和原checkpoint训练来源未完全排除。保留原切分/模型，不擅自重校准。未来独立性主张必须排除所有profile来源并建立实际新条件，不能把剩余69条自动命名最终盲测。禁止查询旧holdout前向/标签。

B1的H17处于部分adapter之后，仅作P2.5分析，不可作为选择专家前的base-only Router输入。跨模型比较锁定共同task/reset/初态SHA/seed/protocol，报告rescue与break两数及任务区间，不拼历史500与当前300。最新开发300：A3=244、B0=260、B1=284、BF16=292、A0=294；历史411/412、430/431分来源保留。

# VLA-Quant experiment admission

## Next boot A1/A2 audit (2026-09-30)

Read [the next-boot audit and runbook](docs/NEXT_BOOT_A1_A2_AUDIT_AND_RUNBOOK_20260930_CN.md) before starting a new instance plan. The previous env_seed=1 condition and old launcher are invalid by default. A1 requires a genuinely observable, reproducible evaluation condition, actual model-input hashes, source/data-overlap audit, task validity, and a protocol-only first-shard gate. Perturbing reused official initial states is development stress evidence, not independent reset replication; cluster such analyses by base reset. A2 fixes global seed and Response-SVD and changes only the separately named distillation sample-order seed after a valid independent A1. Do not use post-divergence H17 as a same-state C0/C3 label. If no admissible condition exists on a paid instance, archive and shut it down rather than idle.

## Future A1/A2 minimal pairing (2026-09-29)

For future A1/A2, run only strictly paired C0 versus C3 on a demonstrably new observable evaluation condition. C1 is excluded; C2/14L and BF16 are separate reference questions, not required replication controls. Record episode-level rescue (C0 fail, C3 success), break (C0 success, C3 fail), their net, task and preregistered within-task context strata, uncertainty, and costs. A2 may change only the C3 training random seed after a valid A1. The old five-configuration 110 run remains immutable historical evidence and `A1_INVALID_CONDITION`. Rescue and break alone do not prove predictable chunk-level contextual routing. Follow [the new protocol](docs/A1_A2_MINIMAL_REPLICATION_AND_CONFLICT_PROTOCOL_20260929_CN.md); P4/P5 remain locked.

## Mandatory evaluation smoke gate (2026-09-29)

Before any long closed-loop evaluation, follow [the evaluation smoke gate](docs/EVALUATION_SMOKE_GATE_20260929_CN.md). A PRE-RUN CARD and source audit must establish the exact evaluator call order, fixed artifact hashes, one changed variable, and non-overlapping data. Run a small no-policy condition probe in that exact order, with same-condition repeats and policy-visible query0/after-eight-step hashes. Then run a paired policy micro-rollout and inspect actual commands, manifests, and traces. Stop at a first-shard gate (at most 50 episodes/configuration) before scheduling further shards. A failed condition or unchanged trace blocks expansion; identical success labels alone do not prove invalidity. A probe PASS is not a long-run approval. On an active paid instance with no admissible next step, archive and close it under the quota protocol. Do not run seed-only A1 again or start A2 until a truly new observable condition passes these gates.

## A1 seed validity gate (2026-09-29)

The 110 run recorded `env_seed=1` and 450 episodes per configuration, but its C0/C1/C2/C3 success labels were identical to the previous 450-episode `env_seed=0` run for every task/reset pair. The official initial-state hashes were also unchanged. Three matching C3 shard boundaries had byte-identical policy traces. The original fixed-action seed probe omitted the evaluator's `seed_all(model_seed)` call after `env.seed`; an evaluator-order probe found identical query0 and subsequent eight-step observation hashes for all ten sampled tasks. Classify this run as `A1_INVALID_CONDITION`, a protocol regression, not independent evaluation replication. Do not launch A2 training, another seed-only GPU repeat, Router, or P4/P5 until a versioned, single-variable evaluation condition demonstrably changes the observations under the actual evaluator order and passes data-overlap audit. See `reports/experiments/p2-shared-peft/20260929-110-a1-envseed1/`. Instance 110 received a native shutdown execution receipt; platform OFF was not independently verified, and its heartbeat is paused.

## Low-frequency continuity for every future experiment (2026-09-29)

Run each approved immutable stage plan through the recoverable server-side sequential supervisor. While a user-opened instance is actively experimenting, use a roughly 15-minute thread heartbeat to read only the supervisor status/revision, runner, and GPU processes. Stay silent when state is unchanged; inspect episode logs only for a stage event, failure, idle GPU with pending work, or quota closure. Do not repeatedly wake the conversation with one-minute sleeps or manual polling to wait for time to pass. A completed stage must be analyzed and the next admissible preregistered plan started within one heartbeat cycle; if none exists, archive and shut down the paid instance. Check the current five-hour quota at stage boundaries and near the closure threshold. Around 15% remaining, finish server/local/GitHub backup and SHA checks; around 10%, stop new experiments and shut down the current instance with at least 3% reserved. Pause the heartbeat after closure so it is silent between runs. A heartbeat does not guarantee quota checks or shutdown when the client is unavailable. Apply this rule to every future instance and session; use the current session's actual host, paths, and shutdown authorization.

Before proposing, coding, launching, or extending any GPU experiment, execute the full [Research Governor](docs/VLA_RESEARCH_GOVERNOR_20260926.txt). This is a required admission gate, including for a resumed or automated stage.

Read the current master plan, experiment log, latest accepted session report, canonical baseline table, and stage gate decision. Record the current primary stage, locked stages, and one main research question. Resolve conflicting results by source audit before using them as a baseline.

Save a pre-run card with hypothesis, one changed variable, fixed control, primary metric, positive and negative decisions, data split, backbone hash, output directory, cost estimate, and stopping condition. Classify each run as contract, diagnostic, development, or final evaluation. Do not run an experiment if either outcome would merely trigger another similar variant.

After each run, save a post-run decision stating what is supported, what remains confounded, whether the stage or branch advances, and the single next question. Apply the project's quota, backup, and shutdown rules independently of scientific admission. When no admissible experiment remains on an active paid instance, archive and close it rather than waiting for user input.

Current primary stage: P2.5 closed-loop alignment diagnosis. Exact 12L backbone, blocks 18–19, rank8 Recovery-LoRA. P4 experts and P5 Router are locked. The 50-episode pilot has no observed 12L-to-14L headroom and cannot yield a W4 recovery fraction. See [P2.5 plan](docs/P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md) for the current evidence and allowed controls. New canonical evidence may change this state only through a versioned gate decision.
