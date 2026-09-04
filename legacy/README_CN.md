# 历史实验代码：默认禁止运行

此目录保留服务器上曾使用过、但已被审计发现存在算法口径或校准覆盖问题的脚本，以便追溯实验。`scripts/install_adapter.sh` 不会安装这些文件。

- `calibrate_scaled_w2.py` / `calibrate_scaled_weight.py`：自定义整行对称 W2，不是官方 AWQ；默认只有三个有效码值。
- `calibrate_scaled_w2_sequential.py`：首次视觉调用即停止，遗漏第二相机。
- `calibrate_awq_action_aware.py`：动作端选 scale，但仍使用上述自定义 W2 量化器，并在同一小校准集选超参数。
- `run_eval_scaled_w2*.py`：只用于读取上述旧 profile。
- 组件和 inventory 脚本：用于历史定位，不代表发布级量化流程。

若必须重跑历史实验，应放在独立环境，输出标记 `legacy-custom-w2`，不得写成“官方 AWQ W2A16”。
