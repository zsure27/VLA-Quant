# 封装审计与修复记录

## 2026-09-04

本次封装没有重写量化算法，也没有产生新的 GPU 成功率。确认并修复以下工程漏洞：

1. 官方函数来源校验：导入 AWQ/SmoothQuant 后检查实际源文件必须位于命令指定目录，避免 `sys.modules` 缓存静默使用其他环境的同名包。
2. 双相机校准覆盖：AWQ 校准不再以全局 256 行提前停止；每次模型调用先抽固定行，全部样本完成后再均匀下采样，使主相机和腕部相机、全部 32 个样本都有机会进入候选集。
3. Profile 精度：AWQ scale/clip 与 SQ activation absmax 改为 FP32 落盘，避免 FP16 的溢出、下溢和额外舍入。
4. Profile 分片一致性：评测前比较每片的格式、方法、bit、checkpoint、样本、group size、alpha 和官方源码 hash。
5. Profile 内容检查：验证实际权重形状及 scale/clip/activation_absmax 的有限性。
6. 检查点与官方源码绑定：拒绝把一个 checkpoint 校准的 profile 应用到另一个路径；拒绝官方源文件 hash 变化。

仍保留但不默认安装的历史问题：

- `legacy/qvla/calibrate_scaled_w2.py` 使用整行对称量化，默认 W2 裁剪网格只有 `{-1,0,1}` 三个有效码值，无 group128。
- `legacy/qvla/calibrate_scaled_w2_sequential.py` 在视觉层第一次调用即中断，双相机输入只采到第一路。
- 历史 W2 的尺度目标与官方 AWQ block/submodule 搜索不一致。
- 当前维护版 AWQ 仍是逐 Linear 搜索；修复采样和校验并不会自动把它变成官方完整 block AWQ。

验证边界：本地完成 Python 语法、Shell 语法、422 target 静态计数与官方五个源文件比对。当前电脑没有对应 PyTorch/CUDA/模型，GPU 数值验证必须在新 AutoDL 实例重新执行。
