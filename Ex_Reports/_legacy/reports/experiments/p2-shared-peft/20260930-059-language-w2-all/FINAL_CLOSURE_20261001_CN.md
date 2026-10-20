# 059 本轮收尾回执

059 的语言扩展 LoRA 正式配对开发评测为：C0 244/300、C3 260/300、LW 284/300、BF16 292/300、全 W4 294/300；LW 相对 C0 有 45 rescue、5 break。P2.5 在 LW 学生访问的 50 回合、678 次观测上完成同观测 BF16 教师诊断，两次 LW 失败回合的夹爪分歧较高，尚不能据此作因果或 Router 可行性结论。细节见本目录的结果报告与 P2.5 报告。

五小时额度在阶段边界实时读取为剩余 73%。本轮提前收尾，是因为下一项视觉 LoRA 或纯 W2 实验尚未完成版本化实现与契约烟雾门禁，不能让 059 长时间空跑；不是额度达到 15%/10% 门限。没有启动 Router/P4/P5，也没有触碰 `offline_final_holdout`。

关机前确认 P2.5 顺序计划的唯一阶段退出码为 0，50 回合、678 次查询完整；`SHA256SUMS.txt` 全部通过，GPU 无计算进程。服务器持久盘与本机忽略目录各存有两个互补归档：

| 归档 | 字节 | SHA256 |
| --- | ---: | --- |
| `20260930-059-language-w2-all-full.tar` | 885104640 | `bb6c1607abc8552cdcc4b041bc44bd20f5dacd3de73d51ddcbf0da0690d232f8` |
| `20260930-059-language-w2-all-p25-delta.tar` | 18227200 | `e93d615082f8d6403a31d2d4a4b05eead693ede99676970e0eba083700bb5797` |

服务器文件所在目录：`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/`；本机副本：`C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/backups/experiments/p2-shared-peft/`。两份归档都在两端核验了 SHA。原生关机核验目录 `/root/autodl-tmp/qvla-repro/backups/059-language-lora-closure-20261001/` 以硬链接包含两份归档和 Git bundle，`SHA256SUMS.txt` 与 `RESULTS_SHA256SUMS.txt` 均通过。

关机前 GitHub 固定仓库 `zsure27/VLA-Quant` 分支 `codex/repository-cleanup-20260922` 的本机与远端提交均为 `47c9bbb6389767b05a3ef178d53b9c6dc0ed78b8`。随后原生工具在预检通过后输出：`time_utc=2026-09-30T21:38:03.517024+00:00`、`supervisor_pid=837`、`execute=true`、`backup=/root/autodl-tmp/qvla-repro/backups/059-language-lora-closure-20261001`、`native_shutdown_sha256=0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`；随后 SSH 被远端关闭。该回执证明已执行容器原生关机请求，**控制台 OFF 与停止计费未独立核验**。本记录提交后应再次核验 GitHub 远端 SHA。
