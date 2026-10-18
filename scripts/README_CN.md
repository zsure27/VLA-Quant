# 运行、恢复与收尾入口

当前没有活动实验：091已原生关机、心跳暂停；B1时间覆盖以正式各250回合收束，旧续片不再恢复。下次用户开启实例后，先依据[近三轮总结与短期计划](../reports/experiments/p2-shared-peft/20261017-three-rounds-summary-and-short-plan/REPORT_CN.md)和[小规模探索策略](../docs/EXPLORATORY_QUANT_FINETUNING_20261017_CN.md)为具体问题选择入口，不批量执行历史脚本。

历史B2/B3工具为`prepare_extended_peft_plans.py`、`train_extended_peft.py`、`prepare_a4_student_state80.py`、`audit_extended_peft_eval.py`与`run_locked_peft_stage.py`。[当时开机卡](../docs/B2_B3_NEXT_BOOT_RUNBOOK_20261008_CN.md)保留来源，不代表这些已完成阶段仍待运行。B1真实评测使用其对应冻结版本和独立artifact身份，不能直接借旧B3 loader代替。

先读根目录 [README](../README.md) 的实验范围。服务器上执行脚本前核对当前实例、GPU、checkpoint、profile、冻结评测器、官方初态和五小时额度；每次使用有限分片和独立输出目录。

每轮实验用 `vla_stage_supervisor.py` 执行不可变计划，只在该轮活跃期间启用约15分钟的心跳。心跳只查状态、实际runner、GPU与磁盘；未变静默，不逐回合轮询。预先约定探索筛查及停止条件，不默认每候选50或450回合；必要实现和数值门禁保留，只验证变化部分。额度15%停止扩展并保存、10%保存终止并确认GPU空闲后原生关当前实例，预留3%；特殊豁免须有当前用户授权。完整归档优先上传GitHub Release并逐项核验，小结果和索引推送Git；不长期保留本机全量副本。关机后暂停心跳，客户端不活跃时不能保证独立读取额度或关机。

| 用途 | 入口 |
|---|---|
| 环境恢复与安装 | `bootstrap_autodl.sh`、`install_adapter.sh`、`repair_install_20260911.sh`、`check_runtime.py` |
| 当前 AWQ 有限闭环 | `run_awq_baseline_shard.sh`、`run_awq_scope_shard.sh`、`run_awq_full_scope_shard.sh`、`run_awq_g64_language_family_shard.sh`、`run_awq_p0_visual_group_shard.sh` |
| 连续阶段执行 | `vla_stage_supervisor.py` 读取不可变JSON计划并无间隙顺序执行；状态只在阶段启动、完成、失败和终态时增加 revision，可用 `--wait-after-revision` 低频等待事件，禁止以逐回合日志轮询代替阶段事件 |
| AWQ/SQ 机制实验 | `run_awq_spatial.sh`、`run_awq_validation.sh`、`run_smoothing_ablation.sh` 及相应 `run_awq_*` 有限诊断脚本 |
| 状态与配对审计 | `awq_scope_status.py`、`baseline_shard_status.py`、`verify_awq_scope_backup_local.py`、`verify_deduplicated_results.py`、`verify_session_manifests.py`、`validate_repository_layout.py` |
| 归档与收尾 | `sync_vla_remote_closure.ps1`、`backup_active_diagnostics.py`、`vla_push_local.ps1`、`close_vla_session.ps1`、`vla_shutdown_remote.py` |
| 私有GitHub归档与按需恢复 | `github_archive_store_v2.py` 上传并下载校验；`publish_private_archive_index.py` 发布小恢复目录；`restore_github_archive.py` 逐块与整包SHA恢复；`remove_verified_local_backups.ps1` 默认只检查，核验发布目录后按精确路径清理。见[恢复说明](../backups/github/RESTORE_CN.md) |
| 本地报告重建 | `build_experiment_summary.py`，只读取已取回数据，不连接服务器 |

目录职责：`qvla/` 保存维护中的模型、量化、LoRA 与评估实现；`diagnostics/` 保存只读机制分析；本目录保存执行、归档与验证入口。历史命令在报告、服务器持久盘和复现清单中按原脚本路径引用，因此保留稳定文件名，按上表选择入口；已停用或失效的命令由原路径的拒绝执行保护拦截。`scripts/verify_deduplicated_results.py` 核验全部 `results/indexes/DEDUPLICATED_FILES_*.json`；`scripts/verify_session_manifests.py --session <会话名>` 可定点核验某轮，省去每次遍历全部会话；`scripts/validate_repository_layout.py` 只解析 Git 跟踪的 JSON，不碰被忽略的完整本机归档。

2026-09-29 的 A1 `env_seed` 单变量已判为 `A1_INVALID_CONDITION`；相关历史入口不得当作独立条件复现实验直接运行。下一次长评测先完成 [`烟雾门禁`](../docs/EVALUATION_SMOKE_GATE_20260929_CN.md)，再解锁长片。

`run-official-w4-smoke20.sh` 和 `run-official-quant-validation.sh` 是特意保留的**拒绝执行入口**。安装时把它们复制到服务器，是为了覆盖可能遗留的旧 v2 自动评估脚本；不得移除这些防误用保护，也不得把它们当作当前测试命令。

实例刚开机并完成 SSH 身份核验后，先运行 `sync_vla_remote_closure.ps1 -ExpectedHostname <实际hostname>`。它把备份和关机辅助脚本上传到持久盘，逐文件比对 SHA256 并执行 Python 编译检查。`close_vla_session.ps1` 在任何备份工作前也会再次执行同一预检，并接受 `-SshHost`、`-SshPort`、`-IdentityFile`、`-KnownHostsFile` 和 `-ExpectedHostname`，不再写死 107 的端口或 known-hosts 文件。

`close_vla_session.ps1` 默认只预览，必须显式传入 `-Execute` 才会按身份、远端工具同步、结果备份、SHA、GitHub认证与推送、原生关机回执的顺序执行。每次显式传入 `-ExperimentModule`；报告和结果路径须为同一模块下的同名 `YYYYMMDD-session`。它是历史本机完整副本流程，不能据此认定已实现Release上传或云端验收。后续按[GitHub备份策略](../docs/GITHUB_BACKUP_POLICY_20261017_CN.md)选择匹配当前会话的归档工具，上传和远端SHA核验后清理临时本机副本。脚本不验证平台控制台OFF或停止计费；回执与平台状态分开表述，不自动开机。

五小时额度剩余约15%时停止扩展并完成归档与GitHub核验，保留具体资产索引和SHA。历史 `-UseVerifiedFallbackOnly -Execute` 仅接受满足该脚本契约的完整服务器归档；它会重新核验SHA、Git bundle、GPU空闲和AutoDL supervisor，再发出原生关机请求。单个阶段包或只有上传回执不能冒充完整收尾备份。

新实例预装示例：

```powershell
./scripts/sync_vla_remote_closure.ps1 `
  -SshPort <端口> `
  -IdentityFile <私钥路径> `
  -KnownHostsFile <known-hosts路径> `
  -ExpectedHostname <实测hostname>
```

实际关机仍使用 `close_vla_session.ps1`，并传入同一组连接参数。脚本不会登录控制台、开机、删除实例或设置定时任务。

`run_awq_p0_visual_group_shard.sh` 固定语言为 G64 attention-only no-clip，并接受 `dino-g64-siglip-g128`、`dino-g128-siglip-g64` 或 `visual-g128` 配方。三格使用相同 checkpoint、初态和 G64 语言 block 坐标，用于隔离两个视觉编码器分支的 group-size 影响。
