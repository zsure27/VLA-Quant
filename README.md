# VLA 快速量化与微调实验备份

安装中遇到 NumPy 2.x 冲突或 `openvla_utils.py SHA256` 不符，请先看 [2026-09-11 安装修复](docs/INSTALL_REPAIR_20260911_CN.md)。在本仓库更新后运行 `bash scripts/repair_install_20260911.sh`，无需删除原环境/数据，也不要绕过校验。

## 2026-09-08 更新：先验证基线，再做可视化

当前为审计后的候选基线，尚未完成新 GPU 验证。请先看：

- [本轮基线审计、种子修复、AutoDL 运行入口](docs/BASELINE_AUDIT_20260908_CN.md)
- [可视化矩阵、attention/模态/projector 修复决策](docs/VISUAL_DIAGNOSIS_PLAN_CN.md)

正式入口改为 `scripts/run_audit.sh`（controls → sq / awq），profile 升级 v3，旧 v2 不可直接复用。LLM 使用官方 AWQ block 搜索并支持 W2，视觉仍明确标记为自定义适配。固定 OFT Transformers fork + SDPA；不能为了画 attention 直接切 eager。以下原有成绩与说明是历史记录，不是本轮修复后的实验结果；旧运行命令以新文档为准。

本仓库是独立的 VLA 快速量化与微调实验工程，依据截至 2026-09-04 的本地代码和 Codex 历史实验日志整理。目标是在更换 AutoDL 算力后，从 GitHub 重新获得实验代码、固定依赖、重建校准数据并继续 AWQ W2A16 与 SmoothQuant W4A4 研究。Triton 不属于本仓库的仿真验证依赖；只有量化方案完整验证通过并进入真机部署时，才从独立 Triton 算子仓库接入 packed 低比特内核。

## 重要边界

- GitHub 中保存的是源码、小型配置、目标层清单、结果摘要和数据来源清单。
- OpenVLA-OFT 权重、LIBERO 原始数据、512 个校准样本、量化 `.pt` profile、原始评测日志未出现在本地；AutoDL 离线时无法恢复这些文件的字节内容，因此未伪造或声称已经上传。
- 模型和可再生数据应从原始来源重建；不可再生 profile/日志在旧服务器恢复后，应按 `data/README_CN.md` 补做离线对象存储或 GitHub Release/LFS 备份。
- 当前 AWQ/SmoothQuant 都是 BF16 算子上的伪量化精度模拟，不是 packed INT2/INT4 推理，不用其运行时间证明低比特加速。

## 当前实验结论

| 方法 | 精度 | LIBERO-Spatial | 证据范围 |
|---|---|---:|---|
| BF16 | W16A16 | 488/500，97.6% | 完整 500 episodes |
| AWQ 适配器 | W4A16 | 491/500，98.2% | 完整 500 episodes |
| SmoothQuant | W4A16 | 19/20，95% | 冒烟测试 |
| SmoothQuant | W4A8 | 20/20，100% | 冒烟测试 |
| SmoothQuant | W8A8 | 20/20，100% | 冒烟测试 |
| SmoothQuant | W4A4 | 0/20，0% | 失败的冒烟测试 |

20 次测试的区间很宽，不能据此宣称 W4A8/W8A8 与 BF16 等价。AWQ W4A16 使用官方底层量化函数，但缩放搜索仍是 OpenVLA 逐 Linear 适配，不是官方完整 Transformer 子模块搜索。

## 两条后续主线

1. 主线：AWQ W2A16。先比较旧三电平 W2、官方 affine group128 RTN W2 与当前逐 Linear 搜索，再实现共享 QKV、gate/up 的官方子模块重构目标。现有 `legacy/` W2 代码不得直接标为官方 AWQ。
2. 次线：SmoothQuant W4A4。先拆出 W4 与 A4、视觉与语言、attention 与 MLP，并做 BF16 projector 输出替换；只有接口替换显著恢复动作时，再训练 projector/小残差层。

## 快速恢复

在新 AutoDL 实例中：

```bash
git clone https://github.com/zsure27/VLA-Quant.git
cd VLA-Quant
bash scripts/bootstrap_autodl.sh
```

默认只下载并固定源码、安装适配代码，不自动下载大模型或创建环境。完整参数：

```bash
INSTALL_ENV=1 DOWNLOAD_MODEL=1 bash scripts/bootstrap_autodl.sh
```

脚本默认根目录为 `/root/autodl-tmp/qvla-repro`，可通过 `ROOT` 修改。模型下载使用公开 Hugging Face 仓库；私有访问令牌只能通过环境或登录工具提供，禁止写入仓库。

恢复后先执行：

```bash
bash scripts/capture_env.sh /root/autodl-tmp/qvla-repro/artifacts/environment-restored
bash diagnostics/run_first_batch.sh
```

第一批诊断使用独立输出目录，不覆盖旧 profile/checkpoint；先检查离线动作与中间特征误差，不直接启动 500 episodes。

## 目录说明

| 路径 | 内容 |
|---|---|
| `qvla/` | 当前维护的校准、量化、动作敏感度和评测适配代码 |
| `diagnostics/` | W/A、视觉/语言、双相机、projector oracle 与逐层误差探针 |
| `legacy/qvla/` | 历史 W2/组件实验，只供追溯，默认不安装 |
| `overlays/openvla-oft/` | 历史服务器使用的三份 OpenVLA-OFT 覆盖文件 |
| `configs/` | 依赖提交、422 个因果连接目标和运行 manifest 模板 |
| `scripts/` | AutoDL 初始化、代码安装、环境记录和评测脚本 |
| `results/` | 根据原始日志记录恢复的结果摘要；不是重新运行结果 |
| `data/` | 缺失大文件的清单、重建与后续备份规范 |
| `docs/` | 中文论文说明、复现指南与 2026-09-02 源码审计 |

更细的逐文件说明见 `文件说明_CN.md`，已修复问题见 `CHANGELOG_CN.md`。

## 安全与复现规则

- 仅加载自己生成的 `.pt` 文件；PyTorch pickle 文件可执行代码，禁止加载不可信 profile。
- 每次实验记录 Git SHA、模型 revision、样本 SHA256、GPU/驱动、Python 包、seed、量化 target 和 attention backend。
- BF16 与量化实验必须使用相同预处理、初始状态、任务顺序和模型代码。
- 校准数据来自训练 split；禁止用正式评估 episode 微调或选超参数。
- 大文件不得普通提交到 Git；单文件超过 90 MiB 时脚本会阻止归档进入提交范围。

## 已知尚未解决的问题

- 论文未公开完整 SmoothQuant W4A4 baseline recipe；本适配器不能宣称精确复现论文基线。
- 尚无完整官方 block-search AWQ W2A16；当前 W2 仅有局部微探针与历史错误实现。
- 新版 AWQ 的视觉 block 搜索需要自行定义，因为官方 AWQ 没有 OpenVLA 双视觉编码器 recipe。
- 数据和旧 profile 的真实 SHA256 要等原 AutoDL 数据盘恢复后补齐。
