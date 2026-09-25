# P3｜静态低比特 backbone（已冻结）

本模块保存形成当前 backbone 的视觉 group、W4 岛、层数裁剪和外推实验。默认不再搜索新的 W4 block 组合。

- 12L：W4 blocks 8–15、20–23，412/500；PEFT/Router 主底座。
- 14L：W4 blocks 8–15、18–23，431/500；blocks 18–19 升 W4 的静态恢复参照。
- 16L：W4 blocks 8–23，434/500；更强 mixed precision 参照。

12L 和 14L 的唯一精度差异是 blocks 18–19 从 W2 升为 W4，对应 +19/500。P2 的问题是保持这两层为 W2，用远小于两层 W4 的 PEFT 成本追回多少收益。
