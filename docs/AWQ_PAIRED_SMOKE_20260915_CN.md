# 2026-09-15 AWQ 闭环配对预实验

实验在 AutoDL 014（实例 `3fbf46b812-2fdd6883`，4090）上完成。仓库开始时为 `852d350`；Python `qvla-oft`，NumPy 1.26.4、robosuite 1.4.0、MuJoCo 3.1.1，`MUJOCO_GL=egl`。模型为 `/root/autodl-tmp/qvla-repro/models/openvla-7b-oft-finetuned-libero-spatial`。LIBERO Spatial 任务 0–9 各一回合；模型和环境种子均为 0，`paired` 协议，中心裁剪、SDPA，动作分块为 8。三组原始 `EPISODE_MANIFEST` 的任务编号、模型种子、环境种子及初始状态 SHA-256 逐项一致。

| 评测配置 | 成功任务 | 成功数 |
| --- | --- | ---: |
| 原始 BF16（无量化） | 0,1,2,3,5,6,7,8,9 | 9/10 |
| 官方来源 AWQ W4A16，完整 422 层，W4 profile | 0,1,2,3,5,6,7,8 | 8/10 |
| AWQ W2A16 当前候选，完整 422 层：语言去 160 处 clip、主视觉 G64、融合视觉 G128 | 无 | 0/10 |

原始结果及复现命令位于持久盘：

- `/root/autodl-tmp/qvla-repro/eval/awq-current-smoke10-20260915-194658/`
- `/root/autodl-tmp/qvla-repro/eval/bf16-paired-smoke10-20260915-201949/`
- `/root/autodl-tmp/qvla-repro/eval/awq-w4-paired-smoke10-20260915-203534/`

每目录含 `command.txt`、`console.log`、`EVAL-*.txt`、`exit-code.txt` 和 `SHA256SUMS.txt`。请以原始 EVAL 文件和状态清单核对。本轮 W2 从前一轮离线 MSE 改良方案直接进入闭环，但所有十个任务失败；因此离线 MSE 不能替代闭环成功率。BF16 与 W4 在相同初始状态下成功，排除了“当前环境普遍无法完成任务”这一解释。W2 的损伤可能来自语言、视觉或两者交互；仅凭这三组实验尚不能定位具体模块或训练方案。每任务只有一回合，不能把 0%、80%、90% 当成总体成功率估计。

下一步优先用同一组十个初始状态分别做 W2 语言和视觉隔离消融，并记录 action/proprio projector 的特征和动作偏差；再按闭环结果决定敏感层 BF16/W4 回退和小参数 PEFT 修复。SQ W4A4 需先通过无量化平滑的数值守恒检查，再做同协议闭环；已有 SQ 0/20 来自旧环境，不能与本轮比较。Contextual Routing 可研究条件化小型路由器选择视觉、语言或动作敏感层的精度/缩放，但应先以静态的模块回退证明哪一类输入状态受损，再避免在十回合上调参过拟合。
