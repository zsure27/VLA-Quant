# 诊断与分析代码

`diagnostics/` 保存局部机制测试和已完成分片的只读分析。脚本输出应写到独立目录，不覆盖原始 profile 或评测结果。

| 职责 | 主要文件 |
|---|---|
| 运行时与数值门槛 | `self_test.py`、`gates.py`、`probe.py` |
| AWQ 敏感性和干预 | `awq_block_audit.py`、`awq_interventions.py`、`compare_w2_clips.py`、`compare_scope_candidates.py`、`low_rank_recovery.py` |
| SQ 平滑数值核验 | `smoothing_selection.py`、`sq_local_equivalence.py`、`fp32_smoothing_pairs.py` |
| 闭环及来源审计 | `baseline_shard_analysis.py`、`awq_scope_analysis.py`、`audit_policy_traces.py`、`audit_oft_adapter.py`、`verify_awq_clip_intervention.py` |
| 可视化与视频索引 | `plot_awq_rollouts.py`、`plot_block_audit.py`、`plot_results.py`、`paired_video_contact_sheets.py`、`export_scope_disagreement_videos.py` |

`run_first_batch.sh` 是已停用的旧入口，会以退出码 2 拒绝运行；保留同名文件是为了防止误用旧 v2 profile。正式阶段入口见 [`scripts/README_CN.md`](../scripts/README_CN.md)。
