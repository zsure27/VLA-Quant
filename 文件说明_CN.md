# 实验代码与文件总说明

## 1. 当前维护代码 `qvla/`

| 文件 | 用途 | 当前状态 |
|---|---|---|
| `action_jacobian_batch.py` | 在动作空间计算各 Linear/Conv 输出通道的量化敏感度代理 | 可运行；仍需用短 rollout 验证排序 |
| `aggregate_action_proxy.py` | 合并分组产生的动作敏感度文件 | 工具 |
| `assign_gates_from_sensitivity.py` | 按代理分数进行逐通道贪心位宽分配 | QVLA 发布逻辑适配 |
| `assign_gates_marginal.py` | 按相邻位宽的边际误差/节省位数分配 | 自建论文式对照 |
| `build_connected_targets.py` | 从 435 个前向目标排除 13 个动作不可达模块，生成 422 个目标 | 已用静态名单固化 |
| `causal_target_inventory.py` | 通过动作输出连接性检查目标模块 | 诊断工具 |
| `forward_target_inventory.py` | 记录实际前向调用的 Linear/Conv 模块 | 诊断工具 |
| `compare_single_batch.py` | 比较两个单批次输出或量化结果 | 诊断工具 |
| `extract_balanced_calibration.py` | 从四套 LIBERO 训练 TFDS 各取 128 个 episode 的一帧 | 生成本项目 512 帧校准集 |
| `official_quant_adapter.py` | 加载官方 AWQ/SQ primitive、量化权重、平滑 Norm/Linear、注册激活 Hook | 当前维护；仍是架构适配 |
| `calibrate_official_quant.py` | 用双图像、语言和 proprio 收集统计，生成 AWQ/SQ profile | 已修复双相机/全部样本覆盖和 FP32 profile |
| `run_eval_official_quant.py` | 验证 profile 后注入预加载模型并调用原 LIBERO evaluator | 已加强分片、形状、源码和 checkpoint 校验 |
| `run_eval_component_hook.py` | 用 Hook 对选定组件做隔离消融 | 诊断工具 |
| `run_eval_component_w.py` | 对选定组件写入伪量化权重做隔离消融 | 诊断工具 |

所有 `qvla/` 文件会由 `install_adapter.sh` 安装到 QVLA checkout。`official_quant_adapter.py` 调用的第三方源码不在本仓库复制，而由 bootstrap 固定 commit 下载。

## 2. 诊断 `diagnostics/`

| 文件 | 用途 |
|---|---|
| `probe.py` | 新进程运行 BF16 teacher、SQ 分范围消融、projector oracle 和 AWQ W2 局部微探针；记录实际导入源码与输入哈希 |
| `self_test.py` | 检查旧 W2 三电平、官方四码 affine W2、SQ 平滑等价性和输入 clone |
| `plot_results.py` | 只从真实 `metrics.json` 绘制视觉深度、LLM 深度和 7 维动作误差 |
| `run_first_batch.sh` | 串联 teacher、W/A、视觉/语言和 projector 对照；不跑完整 rollout |

探针中的 AWQ W2 仅评估 Linear 局部输出 MSE，文件和字段都标明“不是官方 block AWQ”。

## 3. 历史代码 `legacy/qvla/`

这里保存旧 W2、旧逐层校准、旧组件评估及 QVLA 发布脚本的服务器副本。它们不由安装脚本复制。主要用途是解释历史结果和防止后续把旧 profile 误当新实现。每个已知问题见 `legacy/README_CN.md` 和 `CHANGELOG_CN.md`。

## 4. OpenVLA-OFT 覆盖文件 `overlays/`

- `prismatic/extern/hf/modeling_prismatic.py`：支持 OFT 连续动作头、proprio、双图像和动作 hidden-state 返回。
- `experiments/robot/openvla_utils.py`：历史服务器的模型加载/预处理工具。
- `experiments/robot/libero/run_libero_eval.py`：历史服务器的 OFT LIBERO evaluator。

它们来自上游代码的实验覆盖版，保留原英文注释以便与上游逐行比较。安装前会检查上游原文件 SHA256；不匹配就停止，避免覆盖未知版本。

## 5. 脚本 `scripts/`

| 文件 | 用途 |
|---|---|
| `bootstrap_autodl.sh` | 固定四个源码仓库 commit，可选创建环境与下载模型 |
| `install_adapter.sh` | 安装维护代码、覆盖文件并从 422 名单重建 9 个 target shards |
| `capture_env.sh` | 保存 GPU、驱动、包版本和 Git 状态 |
| `inventory_server_data.sh` | 原服务器恢复后，生成大文件大小/SHA256 清单，不修改数据 |
| `run-official-w4-smoke20.sh` | 校准 AWQ/SQ W4 profile 后各跑 20 episodes |
| `run-official-quant-validation.sh` | AWQ 500 episodes 与 SQ 小规模验证的历史编排 |
| `summarize_libero_log.py` | 提取成功数并计算 Wilson 区间 |
| `measure_command.py` | 记录命令耗时和 GPU 显存抽样 |
| `validate_calib_jsonl.py` | 校验发布版 QVLA proxy 的 JSONL 输入 |
| `verify_repository.py` | 检查语法、JSON、422 scope、大文件和常见密钥模式 |

## 6. 配置、结果和数据清单

- `configs/versions.json`：区分已知的历史提交与封装日临时 pin；LIBERO 历史 commit、模型 revision 明确标为未知。
- `configs/qvla-connected-422.txt`：93 个主视觉、105 个融合视觉、224 个语言目标。
- `configs/run_manifest.example.json`：每次新实验必须复制填写。
- `results/official_quant_validation.csv`：从历史日志恢复的汇总，保留样本量、置信区间和耗时。
- `data/README_CN.md`：未能备份的大文件、历史路径、重新生成方法及后续存储要求。
- `requirements-known.txt`：历史指南中确认的关键版本；QVLA 自身依赖仍由固定上游 checkout 安装。

## 7. 推荐恢复顺序

1. 运行 bootstrap，保存首次环境日志。
2. 下载模型后补齐 Hugging Face revision，不直接相信同名目录。
3. 重建校准集并保存 manifest 和所有 `.npz` SHA256。
4. 先运行 diagnostics；若 smooth-only 不能接近 BF16，停止量化实验。
5. 根据 W16A4、vision-only、language-only、projector oracle 决定 SQ 第二轮。
6. AWQ W2 先完成一个 Llama block 的官方子模块搜索对照，再扩展 422 层。
7. 20 episodes 冒烟通过后才运行 500 episodes；packed kernel 性能另立实验记录。
