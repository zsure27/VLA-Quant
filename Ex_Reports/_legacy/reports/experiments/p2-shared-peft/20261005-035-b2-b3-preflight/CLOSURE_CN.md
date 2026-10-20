# 035 收尾核验

本轮仅完成 B2/B3 的物料、数据来源与代码门禁审计，未启动训练或闭环评测。服务器完整回退备份位于 `/root/autodl-tmp/qvla-repro/backups/b2b3-035-preflight-20261005-020010-852d350`；本轮独立证据包位于 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261005-035-b2-b3-preflight/20261005-035-b2-b3-preflight.tar.gz`，SHA256 为 `065f1d7f117c242e117dfa4d0ae997c0b5b4ffcfe53e4cb4b95f17ab7e26432c`。两者均已复制到本机同名 Git 忽略目录；回退备份 10 个文件的 SHA 与结果归档核验通过，独立证据包远近 SHA 相同。

关机前固定 `zsure27/VLA-Quant` 分支远端提交 SHA 为 `468af377a7acd760a120cf5c738c7f8fc01f1920`。GPU 与训练/评测进程为空。2026-10-04 18:02:46 UTC 对当前核验的 035 supervisor PID 837 发出原生关机，工具输出 `execute=true` 回执，随后 SSH 由远端关闭；平台控制台的 OFF/停止计费状态未独立核验。结构化回执见同名 `results` 下的 `closure.json`。本轮没有新实验结果，也没有动用封存留出集。
