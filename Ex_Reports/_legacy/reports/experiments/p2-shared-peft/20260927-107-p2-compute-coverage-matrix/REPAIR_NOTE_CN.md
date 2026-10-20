# B 完成后处理修复与 C 接续

B=16轨迹/1000步的 `probe.py` 已完成 1000 次优化、生成模型状态、manifest、32 个历史回归样本及 `PROBE COMPLETE`。LoRA 状态 SHA256 `cfbcd8a29fe39b430902f9b4aec1ee309b1365bbc154b2429d8c73708335d5db`。原脚本后处理调用裸 `python`，该服务器 PATH 没有此命令，阶段最终 exit127；这不使已完成训练失效。原 `cell-B-preregistered-protocol.json` 中因此有一个空的派生数值字段，保留原件，SHA256 `52f3eec906e9f5d29705af4cbaf968e8126e9c434ac7d7b1a937b8f7e91c5f12`；正确的每轨迹选中帧曝光是 `1000/16=62.5`，由 B 的训练参数独立计算，不假装原字段曾正确记录。

脚本仅将两个后处理处的 `python` 改为现有环境的绝对路径；训练命令、数据、rank、loss、LR 均未改变。C 恢复计划单独版本化，只运行尚未启动的 C=80轨迹/200步，不重复 B。B 的 32 帧仍是回归调试，不作主门禁；完整 A/B/C/D 判断等待 trajectory-level router_dev。
