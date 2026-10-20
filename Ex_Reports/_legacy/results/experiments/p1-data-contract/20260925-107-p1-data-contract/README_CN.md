# 2026-09-25 107：P1 数据与参数化契约

本目录保存进入 shared PEFT 前的机器可读契约证据。默认 backbone 固定为
`awq-w2a16-12l-mixed-spatial-v1`，PEFT 目标只包含语言 blocks 18–19。

## 结论

- LIBERO Spatial RLDS 共 432 条轨迹、52,970 个 transition、10 条指令。
- 以完整轨迹为单位、在每条指令内按稳定 SHA256 排序，得到 298 条 `peft_train`、59 条
  `router_dev`、75 条 `offline_final_holdout`；没有按帧随机切分。
- 原始夹爪动作是 `-1=open、+1=close`。官方 LIBERO transform 执行
  `1-clip(x,0,1)` 后得到 `1=open、0=close`，与 checkpoint 统计一致。
- blocks 18–19 的 14 个 Linear 在 W2/G64 下重建固定 `q/z/Δ` 后，零残差与冻结官方
  fake-quant 权重逐元素完全相等，所有最大绝对误差均为 0。
- 6,324,224 个 scale residual 参数的梯度有限且非零，保存重载逐元素一致。
- rank8 Recovery LoRA 的标准随机 A/零 B 前向严格零输出；base 冻结，梯度与重载契约通过。

## 文件

- `inventory-summary.json`：字段、规模、动作范围、切分数量与泄漏规则。
- `trajectory-split.json`：432 条轨迹的稳定身份和角色。
- `fixed-code-contract.json`：14 个目标 Linear 的 q/z/Δ 哈希和等价检查。
- `peft-train-calibration16-manifest.json`：从 16 条互异 `peft_train` 轨迹选择的 pilot 帧。

Spatial 闭环 states 0–49 已用于历史模型选择，只能作为开发证据。离线 final holdout 在
PEFT 类型、目标层、rank、专家数和 Router 特征冻结前不读取结果；最终闭环须使用改变的
reset/seed 或预注册受控扰动，并保存初始观测哈希。
