# 2026-09-11 安装中断修复

不需要更换服务器、重装系统或删除模型/数据。本次修复针对已保存的安装日志，不代表 GPU 实验已通过。

## 原因一：后续安装升级了 NumPy

旧 bootstrap 只在第一次安装时固定 `numpy==1.26.4`，随后安装 LIBERO/robosuite 的未固定依赖时又选中了 OpenCV 5.0.0.93。该版本在 Python>=3.9 下要求 NumPy>=2，而 TensorFlow 2.15.0 要求 NumPy<2，二者不相容。

修复同时指定 `numpy==1.26.4` 和 `opencv-python==4.10.0.84`，并为 bootstrap 的所有安装阶段设置同一份 `constraints-oft.txt`。不能用 `--no-deps` 强行保留不兼容组合，也不能忽略 pip 的冲突提示。

本轮查询了 PyPI 元数据：OpenCV 4.10.0.84 在 Python 3.10 下的 NumPy 下界是 1.21.2；日志中的 Numba 0.67.0 声明支持 NumPy>=1.22,<2.6，因此本次不额外降级 Numba/llvmlite。其他未锁定包仍需运行 `pip check` 和真实导入测试，不能称为全部依赖已完整锁定。

来源：[OpenCV 5.0.0.93 元数据](https://pypi.org/pypi/opencv-python/5.0.0.93/json)、[OpenCV 4.10.0.84 元数据](https://pypi.org/pypi/opencv-python/4.10.0.84/json)、[Numba 0.67.0 元数据](https://pypi.org/pypi/numba/0.67.0/json)。

## 原因二：维护脚本误用了 Windows 字节哈希

以 `experiments/robot/openvla_utils.py` 为例：

- 旧脚本 expected `6c25918f...` 是 CRLF 换行的哈希。
- 日志 actual `eed754d7...` 是固定 QVLA 提交原始 LF 文件的正确哈希。

本地用固定提交 `26cc4821a3be4c003d09d3c7997b38db2a347982` 的 git blob 重算三份文件，确认是纯换行差异，而不是服务器文件损坏。新安装器仅归一化 CRLF→LF，不忽略其他内容修改；先预检所有覆盖目标，全部通过才备份和复制。已安装的当前维护版本可安全重跑；未知修改仍拒绝覆盖。

## 如何恢复

在本地克隆的 **VLA-Quant 仓库目录**中执行（不要在 `src/QVLA` 上执行 git pull）：

```bash
git pull --ff-only
bash scripts/repair_install_20260911.sh
```

若之前使用非默认 ROOT，继续传同一值：

```bash
ROOT=/原实验目录 bash scripts/repair_install_20260911.sh
```

脚本会激活 qvla-oft 环境、确认 Python 3.10、保存修复前后 pip freeze、联合修复 NumPy/OpenCV、执行 pip check 和 NumPy/OpenCV/Numba/Torch/TF 导入检查、验证 OFT attention 实现，最后继续安装覆盖文件和 qvla 工具。原源码会备份，不清理检查点、样本或 profile。

不要此时重新跑完整 bootstrap：已有 QVLA checkout 可能处于已安装或部分安装状态，干净源码保护会阻止重克隆流程。这个修复入口不需要重克隆，也不跳过文件校验。

成功标记：`DEPENDENCY_IMPORT: PASS`、覆盖文件预检/安装完成、`INSTALL_REPAIR: PASS`。任一步报错会停止；回传第一处错误，不手动删掉检查。

`root user` 警告本身不是此次失败原因，在 root 用户运行的独立 conda 环境中也会出现；实际环境由脚本打印的 Python 路径确认。

本脚本不下载模型/校准数据。修复成功后仍需确认这些文件齐全，再执行 `STAGE=controls bash scripts/run_audit.sh`。CPU 安装器测试已通过，服务器实际依赖导入和完整 GPU 流程仍以运行结果为准。
