# QVLA AWQ基线离线核验结果（2026-10-12）

结论：PASS_ARCHIVED_A0_PROVENANCE_AND_METRICS。此为本机历史来源/指标审计；论文AWQ严格复现、当前GPU数值检查和B5尚未完成。

| 对应Spatial结果 | BF16 | A0/W4 |
|---|---:|---:|
| QVLA表2 | 97.6% | 93.0% |
| 本项目历史500 | 97.4% | 97.2% |
| 历史数据筛除reset0–4，450 | 440/450 | 439/450 |

截图92.5%是四suite平均；纠正口径后仍有4.2个百分点W4差距。完整500与450子集说明“样本从300扩大”不足以解释它。

已核验：A0实际422目标W4/G128、未撤销语言clip、源码SHA与profile来源一致、官方AWQ primitive源码一致、500唯一task/reset配对和0程序异常。未发现漏量化或BF16误标证据。语言采用官方block方法，视觉自写适配，校准32观测；历史paired逐回合种子也与公开upstream流程不同。

公开QVLA仓库没有表2 AWQ完整实现/校准配置。Appendix F的512轨迹用于QVLA敏感度，不应强行当AWQ预算。不能通过人为调差模型来匹配93%。完整差异和下一轮可核验桥接见：
[对齐方案](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/docs/QVLA_AWQ_ALIGNMENT_AND_B5_SPATIAL_20261012_CN.md)。

最初使用Python3.12标准库复核来源和历史CSV。随后发现本机已有隔离CPU环境（torch 2.14.0+cpu、transformers 4.40.1），实际执行 `tests/test_contracts.py`：9项通过，涵盖422目标、checkpoint内容身份、张量克隆、官方语言AWQ缩放等价、拒绝不合规profile及注意力/观测契约。首次运行两项因Windows临时目录权限失败，指定工作区临时目录后全部通过。仍没有实机完整模型张量/GPU复测；对应初始回执明确没有重新读取profile二进制。
[CPU回执](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/results/experiments/p0-foundation-baselines/20261012-qvla-awq-alignment/cpu-audit.json)
[审计脚本](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/scripts/audit_qvla_awq_alignment_20261012.py)

用户已授权基线处理后探索Spatial B5；reset5–49为450/模型的扩大同任务开发验证，非盲测。四主配置同条件重新评测；如路由选择B3/B4，则内部保留B3最佳固定控制。当前H17在前层adapter之后，不得用来选择整套专家。
[B5 PRE-RUN CARD](C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/reports/experiments/p5-contextual-routing/20261012-b5-spatial-preparation/PRE_RUN_CARD_CN.md)

091由用户无卡开启，已通过用户提供的可信ED25519指纹连接，核对hostname；torch CUDA不可用，cgroup仅2GiB内存和0.5 CPU。收尾helper已同步、SHA及编译通过。本轮没有读取封存标签、训练router或产生新的闭环成功率。

2026-10-12 14:16前实时五小时剩余14%，按用户最新覆盖进入刷新前进度保存，不关闭091。预期北京时间15:16刷新必须以实时接口确认；刷新后继续实际profile/checkpoint内容审计、真实随机流CPU探针和B5实现/计划。保存不等于基线差距已解释或B5已可运行。

