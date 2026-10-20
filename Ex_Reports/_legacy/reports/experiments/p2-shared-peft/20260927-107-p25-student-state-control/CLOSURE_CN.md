# 107 本轮收尾回执

2026-09-27 16:37:47 UTC，确认矩阵与同观测对照进程均已终止、GPU 空闲后，调用107持久盘上的 `vla_shutdown_remote.py --execute`。收到原生执行回执：`supervisor_pid=838`，`execute=true`，`native_shutdown_sha256=0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`，备份校验目录 `/root/autodl-tmp/qvla-repro/backups/closure_20260928_p25`；随即远端主动断开 SSH。平台控制台 OFF 与停止计费状态未独立核验。

服务器持久盘及本机三个完整 tar 的 SHA256 均核对通过，见 `BACKUP_MANIFEST_CN.md`；小型原始指标、分析与代码已推送固定仓库 `zsure27/VLA-Quant` 的 `codex/repository-cleanup-20260922`，关机前远端 SHA 为 `783a1a5ab6057ee005db6e848c6b6bbb01a5cfc6`。后续补充本回执的提交应以 GitHub 远端读取值为准。本轮低频心跳已暂停；本次没有本地电脑关机授权。
