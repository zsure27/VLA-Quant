# 046 B2/B3 阶段审计（2026-10-08，额度刷新前）

状态：B2 v2 在五配置各 10 回合微测后停于 `audit-micro`，首片 50 回合未启动；B3 未启动。旧产物保留作诊断，不计作通过门禁的 B2 结果。

## 失败定位

- 评测 trace 在 `get_action` 前保存原始 8D proprio，字段 `state_space=raw_proprio_before_get_action`。
- 冻结 OFT `get_vla_action` 对 `obs["state"]` 就地赋值为 `normalize_proprio` 结果。随后 `on_policy_capture.record_query` 保存的 NPZ `state` 因而是策略实际使用的归一化 proprio。
- 旧 `audit_extended_peft_eval.py` 直接比较原始 trace 与归一化 NPZ，导致全部微测完成后错误地失败。更重要的是，新 B2 训练入口经 `prepare_inputs` 对 NPZ 再次调用 `normalize_proprio`，使冻结 BF16 教师查询、学生训练和真实部署输入不一致。历史 B1 的 `student-state80` 使用同一捕获路径，存在相同风险；历史结果不因此被改写，但机制解释须降级并单独审计。
- 旧 B2 v2 的 1000 步训练与五配置微测是诊断数据，禁止用于放行长片或声称视觉增量 LoRA 有闭环收益。旧 B2 锁定 checkout 为 `8bb02a4dffb89b5357f93c6aa789e109d32f7ef9`；B3 空间预检修正为 `8c15e0242784dece615aa21489ccea9a9d8af954`。

## 修复门禁

新版本须显式标记训练观测的 proprio 空间，对已经归一化的学生状态直接传入模型；原始 peft_train 校准帧仍只归一化一次。审计器用同一冻结 checkpoint 的 q01/q99 规则把 raw trace 归一化后，与 NPZ 逐查询比较，同时保留图像、任务、8 步 chunk、动作、manifest、SHA 与严格配对检查。B2 和 B3 分别新建不可变计划、重新做 10 步烟雾与微测，协议门禁通过后才能做 50 回合首片。不得覆盖旧产物。

## 当前备份

服务器：`/root/autodl-tmp/qvla-repro/backups/transfer/20261008-046-b2v2-failed-micro.tar.gz`；本机：`backups/experiments/p2-shared-peft/20261008-046-b2-b3/20261008-046-b2v2-failed-micro.tar.gz`。两份 SHA256 均为 `6d37bac8655c4db10b45b88c64a4ab7e4f7a7ff0b7c00f686e926fe6168926e1`。归档包含 B2 v2 原始训练、五配置微测与控制状态；未包含尚未开始的 B3。服务器仍在线，本报告不是最终收尾回执。

## 限制和后续

评测 reset20 起是复用开发条件，不是独立 A1。微测与首片重叠，不能相加。待修复完成后先重跑小门禁，依据协议有效性而非成功率决定是否扩展；若无法安全修复，保留原始产物并按当前额度与实例收尾约定处理。
