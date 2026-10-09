# A/B模型规范命名与审计入口

> 2026-10-09覆盖：046开发reset20–39每配置200已完成，B2=190、B3=192，B1=194、A4=30、BF16=194、A0=195；协议通过，完整本机归档仍待补传。新增B4=A4＋视觉196/语言224个Linear rank8联合1000步，独立registry扩展`configs/model_registry_b4_v1.json`，CPU通过/GPU待测。旧registry字节保留以保护历史artifact。执行见[B4 runbook](B4_NEXT_BOOT_RUNBOOK_20261009_CN.md)。以下20261005的HOLD是历史状态。

完整四列表格、数据划分/泄露发现、代码修复、配对指标与后续硬门禁统一维护在 [2026-10-05审计报告](../reports/experiments/p2-shared-peft/20261005-model-registry-audit/README_CN.md)。

机器可读身份见 [model_registry_v1.json](../configs/model_registry_v1.json)，原始证据SHA及重算结果见 [audit.json](../results/experiments/p2-shared-peft/20261005-model-registry-audit/audit.json)。

规范名称为BF16、A0=W4、A1=16L、A2=14L、A3=12L/旧C0、A4=全合格W2；B0=旧C3、B1=旧LW、B2=视觉增量、B3=全W2＋新语言LoRA。B2/B3仍HOLD_CODE_SMOKE。旧评测阶段A1/A2写作EVAL_REPLICATION/TRAIN_ORDER_REPLICATION。

历史W4校准暴露后续75条PEFT留出中的6条，不能称全流程holdout。训练/测试来源、继承SVD校准80帧的未决来源证据及B1 post-adapter H17限制见完整报告。
