# 2026-09-23 107：完整 W2 视觉 group 归因

## 问题与固定条件

在完整 422 个连接量化目标 W2A16 下，固定语言为 G64、仅 attention 取消额外 clip、MLP 保留 clip；只切换 DINO 主视觉支路和 SigLIP 融合支路的 group size。评测使用 LIBERO Spatial、官方初态 5–9、每任务 5 回合，共 50 回合，种子与 manifest 严格配对。

## 结果

| DINO | SigLIP | 成功回合 | 来源 |
|---:|---:|---:|---|
| G128 | G128 | 3/50 | 本轮 |
| G128 | G64 | 1/50 | 本轮 |
| G64 | G128 | 10/50 | 本轮 |
| G64 | G64 | 11/50 | 2026-09-18 历史配对分片 |

四格 50 个 episode manifest 完全一致。DINO 从 G128 改为 G64，在 SigLIP G128 下增加 7/50，在 SigLIP G64 下增加 10/50；SigLIP 从 G64 改为 G128，在 DINO G128 下增加 2/50，在 DINO G64 下减少 1/50。当前开发分片支持的主要效应来自 **DINO G64**；SigLIP group 的差异较小且方向不稳定。DINO64/SigLIP128 的 10 次成功包含 G128/G128 的全部 3 次成功，另恢复 7 次。

这仍不是可用的完整 W2：最佳仅 11/50，而同初态仅语言 G64 attention no-clip 为 29/50、视觉仅 DINO64/SigLIP128 的大样本结果为 460/500。下一步应固定 DINO G64，不再扩写同一视觉 group 网格；优先定位语言 W2 与全模型组合时的交互，并验证最小 W4 语言保护或静态 PEFT。

## 完整性与异常

- 本轮三格分别为 10/50、1/50、3/50；策略查询分别为 1253、1386、1360，动作块和 proprio 全部 finite。
- 后两格 wrapper 退出 0 并写出 `EXECUTION_COMPLETE`。首格评测器已明确输出 50 回合与 10 次成功，但启动脚本在评测后退出 127：运行中更新了同一路径脚本，Bash 继续读取时在第 50 行把 `0` 当作命令。原始退出码保留；`posthoc-rollout-audit.json` 记录终止计数、查询检查和日志 SHA。后续两格使用冻结只读脚本副本，未复现该问题。
- [配对逐回合数据](data/paired-results.json)标为 `measured_paired_closed_loop`。本轮仍是 BF16 权重承载的 fake quant，不代表 packed INT2 速度或压缩率。
- 服务器持久盘包与本地取回包 SHA256 均为 `3002964800b17c9b5090bf148981c13883083bc485fda312109b124d8e820607`；服务器还保留一份字节相同副本。视频位于服务器 rollout 目录，不在 6 MB 小结果包中。

## 复现分析

```text
python diagnostics/analyze_awq_p0_groups.py \
  results/107-awq-p0-20260923 \
  reports/sessions/20260923-107-awq-p0-visual-groups/data/paired-results.json \
  --both-g64-run results/107-awq-continuation-20260918/awq-scope-shard-full-w2-g64-attention-no-clip-05-09-20260918-032153-93284
```
