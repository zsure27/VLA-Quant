# 046 B2/B3 本轮收尾记录

日期：2026-10-09。用户确认已手动关闭 046；没有取得由本工具执行的原生关机回执，也没有独立核验平台 OFF 或停止计费。本轮不再连接或启动 046，本地 Windows 电脑保持运行。实验低频心跳 `vla-046-b2-b3-experiment` 已设为 PAUSED。

## 结果与解释

严格配对的官方开发 reset20–39，每配置 200 回合，微测与正式片重叠且没有重复相加。A3 164/200，B1 194/200，B2 190/200，A4 30/200，B3 192/200，BF16 194/200，全 W4 A0 195/200。B2 相对 B1 为 3 rescue、7 break，净 −4；B3 相对 A4 为 162 rescue、0 break，净 +162。25–29 和 30–39 的协议硬门禁均通过。B2 的视觉增量 LoRA 未表现出稳定收益；B3 的全 W2 语言 LoRA 在这些开发 reset 上有强恢复，但这不是独立复现，亦不能证明与 BF16 行为等价或放行 Router/P4/P5。详细逐任务、置信区间、成本和限制见 [配对分析](DEV_BOUNDARY_30_39_AND_COMBINED_ANALYSIS_CN.md)。

## 归档状态

- 服务器持久盘归档：`/root/autodl-tmp/qvla-repro/backups/b2b3-046-final-20261009-0001`。关机前归档脚本返回 `PASS_PERSISTENT_ARCHIVE`，核验 `SHA256SUMS.txt` 13 项和 `RESULTS_SHA256SUMS.txt` 18494 项。归档用硬链接组织原始数据，不能算与原目录物理独立的第二份。B3 可再生 SVD spool 留在服务器原路径，没有包含在传输归档中。
- 本机目标目录：`C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/backups/experiments/p2-shared-peft/20261008-046-b2-b3/b2b3-046-final-20261009-0001`。传输中止后仅有 1285 个文件，两份 SHA 清单均缺失，`B2B3-essential.tar.gz` 仅 209715200 字节。已标记 `INCOMPLETE_DO_NOT_USE.txt`，**本机副本尚未完成、不能声称双份备份或本机 SHA 通过**。
- 小型结果、协议门禁与本报告保存在同名 `reports/`、`results/` 目录，推送到固定仓库 `zsure27/VLA-Quant`。关机前的远端提交 SHA 为 `9e9ce6d4220053b68f4969e4bc0e238a46329b42`；本收尾记录将随下一次提交再核对远端 SHA。

## 后续恢复

下次由用户主动开启含同一持久盘的实例后，先核对实例身份和服务器归档清单，补传完整归档到本机目录，再运行 `scripts/verify_vla_archive_local.py` 逐文件校验。校验通过前保留服务器原始数据和本机部分文件，不把本轮记录为完整双份备份。40–49 未运行；关机前持久盘仅约 3 GB 空闲，不应直接追加长评测。
