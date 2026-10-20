# 2026-09-25 107：P2 shared PEFT 机器可读结果

本目录只提交紧凑指标、配置审计和哈希。逐样本特征、完整控制台日志及 adapter state
保存在服务器持久盘：

`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260925-107-p2-shared-peft/`

名称以 `exact12l-` 开头的运行才使用正式语言 W2/G64 backbone。此前无该前缀的运行使用
了语言 W2/G128，只保留为配置错误的诊断，不参与 gate。
