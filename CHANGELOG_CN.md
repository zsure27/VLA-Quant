# 封装审计与修复记录

## 2026-09-25

- 冻结粗粒度 W4 block 组合搜索，固定 16L/14L/12L 的参照角色；12L 成为 blocks18–19 PEFT 恢复底座。
- 将 12L backbone 固化为版本化配置 `awq-w2a16-12l-mixed-spatial-v1`；AWQ-W2A16 为主线，Router 用于恢复纯 2bit 能力，SQ 作为验证后的独立扩展。
- 将报告、Git 小结果和本地完整归档统一为 `<module>/YYYYMMDD-<host>-<experiment>`，并按 P0–P5/次线模块归档。
- 新增 PEFT/Contextual Routing 强制执行协议：先训练数据与 q/z/Δ 契约，再 shared PEFT、专家互补和因果 Top-1 Router。
- 将 Spatial states0–49 标为历史/开发证据，要求模型选择完成前预注册真正留出的最终 reset/seed/扰动协议。
- 明确 Response-SVD 仅是 LoRA 初始化；不同静态量化配置不得直接充当轻量运行时专家。

## 2026-09-11

- 修复跨平台 overlay 哈希错误：旧 expected 来自 Windows CRLF；按固定 QVLA git blob 验证 LF 后统一换行再比较。
- 覆盖安装改为全量预检→备份→复制；重复安装与半安装可恢复，未知修改仍拒绝覆盖。3 项安装器回归测试通过。
- 增加各 pip 安装阶段共用的约束文件，防止 OpenCV 5 再次升级到 NumPy 2，破坏 TensorFlow 2.15。
- 提供不重克隆、不删除模型/数据的修复入口，包含依赖、导入与 OFT runtime 检查。目标服务器结果仍待用户执行回传。

## 2026-09-08

- 固定 OFT 专用 Transformers commit、真实源码 hash 与双向 SDPA 语义检查，避免 eager/普通 Transformers 改变基线。
- 精确校验 422 个模块名；v3 profile 绑定完整权重、样本、实现与官方源码指纹；旧 v2 正式入口停用。
- 新增官方 Llama block AWQ 搜索与 W2 独立 profile，保留真实多模态输入/逐帧 attention 上下文；视觉适配不冒称原生官方支持。
- 相机/帧配额与逐帧调用数校验；SQ 非预期训练梯度截断保护；夹爪分歧改为实际开合离散决策。
- 区分上游环境种子协议与配对回合协议，记录 init-state 哈希；新增单 GPU、TF/RLDS 播种训练入口。
- 模型加载不再自动改写本地 checkpoint；安装维护代码前备份已有 qvla 文件。
- 新增分阶段门槛、AWQ/SQ 全路径离线诊断、可选无后端切换的 attention 概率采样、中文决策方案。
- CPU 数值/回归/双向 attention 检查通过；完整 VLA GPU 校准、评估与训练仍待 AutoDL 验证。详见新审计文档。
- 回归测试捕获并修复 qvla namespace 路径污染：新增显式包边界，避免从相邻旧目录导入旧评估器。9 项 CPU 测试通过。

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
