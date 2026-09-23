# 每轮实验独立报告与可视化归档

最新实测：[2026-09-23 完整 W2 视觉 group 归因](20260923-107-awq-p0-visual-groups/README_CN.md)，固定语言配置后完成 DINO/SigLIP 的 2×2 配对网格。

2026-09-22 的仓库整理仅移除经 SHA256 确认的字节相同文件。当前 `manifest.json` 索引整理后的 Git 文件树；每轮 `backup/SHA256SUMS.txt` 等历史回执仍对应服务器归档时的原目录，不改写。旧路径与规范副本的逐项映射见[精确去重清单](../../results/DEDUPLICATED_FILES_20260922.json)，从清理前 Git 提交也可恢复旧布局。

备份约定：每次实验会话收尾时，将图、对应数据和分析单独备份到 GitHub，便于查验及学术汇报。固定位置为 `reports/sessions/YYYYMMDD-SESSION/`，原始小文件放 `results/SESSION/` 并互相引用。不依赖定时对话或服务器独立看门狗。

```text
reports/sessions/YYYYMMDD-SESSION/
  README_CN.md       # 实测结论、失败、局限、问题解释及下一测试
  figures/           # PNG + SVG；视频取代表帧并保留原片清单
  data/              # CSV/JSON逐样本或逐episode；未测空值，含evidence_class
  manifest.json      # 来源相对路径/hash、生成文件/hash、运行/分析版本
  reproduce.py       # 可从已备份数据再生图表；也可引用固定公共脚本
results/SESSION/
  command.txt / console.log / exit-code.txt / status.json
  config.json / source-version.json / metrics.json / episode-manifest.json
  LARGE_FILES_NOT_IN_GIT.json
```

这是目录规范，不是声称每轮已经有所有指标。没有GPU前向的分析对话，标`local_evidence_review_no_new_gpu_run`；有旧摘要则标`archival_summary`；预实验未跑标`planned`。不得生成假逐样本记录或给未测显存/速度填0。

每张图caption说明模型、量化范围、样本单位、teacher/GT区别、误差线算法及样本复用。闭环分成功矩阵/配对差/失败回放；离线分误差分布/动作维度/clip饱和/层或token统计；训练分loss、有效梯度、整数码冻结、成本；Router分互补矩阵、等预算共享对照与切换开销。按测试问题选图，不机械要求全部。

收尾核对本轮命令、原始小数据、图、分析、环境和恢复说明，生成hash后push `zsure27/VLA-Quant`，验证认证身份及remote commit/文件清单。大weights/profile/校准原件保留持久盘，提交清单/hash/恢复路径，并明确它们未在普通Git异地备份。没有第二份原件不能声称全部数据完整异地备份。

GitHub失败时保留全部待备份资产，不删除唯一结果；仍优先按本会话最新责任关机。清理只删除确认可再生或已核对备份的冗余，先检查完整路径。关机确认后报告本轮成果、问题、待推送项及最新实例；不销毁实例数据。
