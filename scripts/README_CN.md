# 运行、恢复与收尾入口

先读根目录 [README](../README.md) 的实验范围。服务器上执行脚本前核对当前实例、GPU、checkpoint、profile、冻结评测器、官方初态和五小时额度；每次使用有限分片和独立输出目录。

以后每轮实验默认用 `vla_stage_supervisor.py` 执行不可变顺序计划，并只在该轮活跃期间启用约 15 分钟的对话心跳。心跳只查 `status.json`/`--status`、runner 和 GPU；状态不变时静默，不用每分钟 `sleep` 唤醒对话计时，也不轮询逐回合日志。阶段事件、失败、GPU 空闲而计划未完成时及时深入处理；有合规下一计划须在一个心跳周期内接续，无可执行项则分析备份并关当前实例。五小时额度约 15% 完成服务器、本机、报告与 GitHub/SHA 备份，约 10% 停止新实验并关机，至少预留 3%。关机后暂停本轮心跳；新实例必须重新绑定实际身份和路径。客户端不活跃时心跳不能保证独立读取额度或关机。

| 用途 | 入口 |
|---|---|
| 环境恢复与安装 | `bootstrap_autodl.sh`、`install_adapter.sh`、`repair_install_20260911.sh`、`check_runtime.py` |
| 当前 AWQ 有限闭环 | `run_awq_baseline_shard.sh`、`run_awq_scope_shard.sh`、`run_awq_full_scope_shard.sh`、`run_awq_g64_language_family_shard.sh`、`run_awq_p0_visual_group_shard.sh` |
| 连续阶段执行 | `vla_stage_supervisor.py` 读取不可变JSON计划并无间隙顺序执行；状态只在阶段启动、完成、失败和终态时增加 revision，可用 `--wait-after-revision` 低频等待事件，禁止以逐回合日志轮询代替阶段事件 |
| AWQ/SQ 机制实验 | `run_awq_spatial.sh`、`run_awq_validation.sh`、`run_smoothing_ablation.sh` 及相应 `run_awq_*` 有限诊断脚本 |
| 状态与配对审计 | `awq_scope_status.py`、`baseline_shard_status.py`、`verify_awq_scope_backup_local.py`、`verify_deduplicated_results.py`、`verify_session_manifests.py`、`validate_repository_layout.py` |
| 归档与收尾 | `sync_vla_remote_closure.ps1`、`backup_active_diagnostics.py`、`vla_push_local.ps1`、`close_vla_session.ps1`、`vla_shutdown_remote.py` |
| 本地报告重建 | `build_experiment_summary.py`，只读取已取回数据，不连接服务器 |

目录职责：`qvla/` 保存维护中的模型、量化、LoRA 与评估实现；`diagnostics/` 保存只读机制分析；本目录保存执行、归档与验证入口。历史命令在报告、服务器持久盘和复现清单中按原脚本路径引用，因此保留稳定文件名，按上表选择入口；已停用或失效的命令由原路径的拒绝执行保护拦截。`scripts/verify_deduplicated_results.py` 核验全部 `results/indexes/DEDUPLICATED_FILES_*.json`；`scripts/verify_session_manifests.py --session <会话名>` 可定点核验某轮，省去每次遍历全部会话；`scripts/validate_repository_layout.py` 只解析 Git 跟踪的 JSON，不碰被忽略的完整本机归档。

2026-09-29 的 A1 `env_seed` 单变量已判为 `A1_INVALID_CONDITION`；相关历史入口不得当作独立条件复现实验直接运行。下一次长评测先完成 [`烟雾门禁`](../docs/EVALUATION_SMOKE_GATE_20260929_CN.md)，再解锁长片。

`run-official-w4-smoke20.sh` 和 `run-official-quant-validation.sh` 是特意保留的**拒绝执行入口**。安装时把它们复制到服务器，是为了覆盖可能遗留的旧 v2 自动评估脚本；不得移除这些防误用保护，也不得把它们当作当前测试命令。

实例刚开机并完成 SSH 身份核验后，先运行 `sync_vla_remote_closure.ps1 -ExpectedHostname <实际hostname>`。它把备份和关机辅助脚本上传到持久盘，逐文件比对 SHA256 并执行 Python 编译检查。`close_vla_session.ps1` 在任何备份工作前也会再次执行同一预检，并接受 `-SshHost`、`-SshPort`、`-IdentityFile`、`-KnownHostsFile` 和 `-ExpectedHostname`，不再写死 107 的端口或 known-hosts 文件。

`close_vla_session.ps1` 默认只预览，必须显式传入 `-Execute` 才会按身份、远端工具同步、结果备份、SHA、GitHub 认证与推送、原生关机回执的顺序执行。每次还必须显式传入 `-ExperimentModule`；报告和结果路径须为同一模块下的同名 `YYYYMMDD-session`。完整本地归档统一写入 `backups/experiments/<module>/<session>/`。脚本不验证平台控制台 OFF 或停止计费；回执与平台状态必须分开表述。当前只依据用户已开启实例的会话进行收尾，不设置定时关机或自动开机。

五小时额度剩余约 15% 时完成持久盘归档、本机副本和 GitHub 推送，并保留具体备份目录。剩余 10% 时可传入该目录及 `-UseVerifiedFallbackOnly -Execute`；脚本会跳过新的归档、报告重建和本机传输，只同步两份小型收尾工具，并由远端关机工具重新核验归档 SHA、Git bundle、GPU 空闲和 AutoDL supervisor 后发出原生关机请求。

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
