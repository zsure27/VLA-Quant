# 059 语言 W2 扩展 LoRA：首片轨迹门禁修正

## 发现

不可变 `runner-first50-v2-plan.json` 已完成五配置各 50 回合，退出码均为 0，原始产物保留在服务器持久盘 `eval-first50/`。按预注册的硬门禁核验时，实际 `command.txt` 显示 `--trace-actions`，但没有 `--trace-observations`；目录中也无 `on-policy-events.jsonl` 和策略可见观测文件。脚本仅在 `micro` 分片开启该选项。因而新增 reset21–24 的首次策略观测无法逐回合核对。此为记录协议缺口，原 50 回合/配置保留为开发观察，不能放行 `runner-remaining250-v2-plan.json`，也不将其与后续重跑作为独立样本叠加。

## 单一修正与重新注册

新增 `scripts/run_20260930_059_language_w2_all_v3.sh`，仅对 `first50trace:20:5` 阶段增加 `--trace-observations`，输出改为独立的 `eval-first50trace/`，不覆盖任何旧结果。五配置、模型/adapter、exact-12L、初态、随机流、动作语义及每配置 50 回合均固定。新顺序计划为 `results/experiments/p2-shared-peft/20260930-059-language-w2-all/runner-first50trace-v3-plan.json`。完成后用 `scripts/audit_20260930_059_language_micro.py --phase first50trace` 检查实际命令、输入/产物 SHA、250 个严格配对 manifest、每回合首个策略观测、8 步 chunk、7D 有限动作与退出码。门禁只由协议有效性决定，不依据成功率方向筛选。

这一重复评测用于修补观测追踪，不提供新的独立环境样本；如与原首片结果不同，将优先审计可重复性，再决定是否继续。

## 资源与边界

启动修正片前，059 数据盘剩余约 12 GB，当前 Codex 五小时额度剩余 86%。原始 v2 产物仍留服务器持久盘，不删除；新观测文件预估会增加数 GB。若存储不足或协议再次失败，停止扩展、保存分析与双份备份，并按本轮额度约定关闭 059。纯 W2、视觉 LoRA、Router/P4/P5 均不在本次修正范围。
