# 014：A1 新 reset 条件前置核验

## 本轮要回答的问题

固定 exact-12L、C0 与 C3 student-state80 rank8 Recovery-LoRA 后，C3 在真正不同的初始观测上是否仍有闭环收益？[110 的 `env_seed=1` 运行](../20260929-110-a1-envseed1/README_CN.md)在旧 450 回合上得到 C3=389、C0=368、rescue=35、break=14，但实际观测与旧环境 seed 相同，属于协议回归，**不能当独立复现**。因此本轮先执行 [长评测烟雾门禁](../../../../docs/EVALUATION_SMOKE_GATE_20260929_CN.md)，而不是直接重复长闭环。

## 014 物料与条件审计

- 已核验主机 `autodl-container-3fbf46b812-2fdd6883`、RTX 4090、冻结评测器 SHA256 `fe37e8097c286e1946e5194a1f817a2b2362d167a6a5a49443c5aa20314d8873`，C3 adapter SHA256 `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a`，三份 AWQ profile SHA 见 [PRE-RUN CARD](PRE_RUN_CARD_CN.md)。
- 当前 LIBERO Spatial 每任务的 `.pruned_init` 为 50 个唯一状态；同名 `.init` 为 100 个，含现用 50 个及 50 个额外状态。10 任务共 500 个额外状态，但**其筛除原因、与训练示范的关系和物理可解性尚不明**。抽取每任务一个做重复无模型检查：10/10 的原始观测有限且重复一致；两路处理后图像与 proprio 在 10/10 任务上不同。其余 490 个未逐个核验。完整数据在同名 `results` 目录。
- 因 `.init` 的来源无法支撑独立 A1，另从同任务模拟器用预登记种子 `2026093000 + task_id` 生成每任务 2 个**前瞻性** reset，立即冻结为 `.npy`。20/20 有限、各任务内不同、与官方 50 及未剪枝 100 状态无完整状态 SHA 重复；初始和等待 10 步后均未已达成目标。
- 对每任务首个新 reset，按实际顺序重复两次：官方与新条件的两路处理后图像和 proprio 在 10/10 任务上均不同；相同条件自身重复一致。冻结处理器进一步中心裁剪与归一化后，两路像素张量在 10/10 任务上仍不同。**这是条件进入策略输入的证据，不是闭环收益或任务可解性的证据。**
- RLDS 的既有 432 条轨迹清单划分为 `peft_train=298`、`router_dev=59`、`offline_final_holdout=75`。现有训练记录未保存完整模拟器 reset state；本轮没有读取 holdout 图像/标签。新状态在 C3 固定后生成，且与已存官方状态数组精确去重，但不能宣称与训练示范的全部物理状态已完成绝对排重。

## 当前 gate 与下一步

`A1_GATE=HOLD_CONDITION`。本轮预注册先用冻结 BF16 对新条件 index 0/任务做 10 回合任务有效性微型诊断；其结果不用于从两条候选里挑选有利于 C3 的状态。**该阶段未进入任何回合，退出码 2。**启动脚本没有把冻结 OFT overlay 放入 `PYTHONPATH`，实际导入旧版评测入口；模型加载后在解析 `--initial_state_offset 0` 时被拒，`policy-queries.jsonl` 为 0 字节、GPU 已空闲。这是环境入口错误，不能记为 BF16 的任务失败或成功。原命令、两份相同的物料 SHA、控制台错误和退出码均已保留在服务器 `bf16-failed-attempt/`。

已把冻结 OFT overlay 加入未来重跑脚本的 `PYTHONPATH`，并以 CPU 独立断言实际导入路径为 `/root/autodl-tmp/qvla-repro/overlays/awq-p0-stage-20260923/oft/experiments/robot/libero/run_libero_eval.py`，其配置确含 `initial_state_offset`；断言通过。修复脚本保留为 `run_bf16_validity_retry1.sh`，**本轮未重跑**。五小时额度于门限检查时仅剩 14%，遂按约定停止扩展并归档。下次开机可从同一冻结状态、同一 10 回合诊断开始，仍须核对当轮实例及实时额度。

只有后续任务有效性、真实命令/文件 SHA、query0 输入与动作 trace 全部核验，才会运行 C0/C3 每任务 1 个 reset 的严格配对微型闭环。微型闭环合格后，仍须预注册独立条件、完整样本量、配对区间和成本，首片最多 50/配置设硬门禁；未获放行不排队长片。不得将离线差异、BF16 成功率或旧 `env_seed` 数值变化写成 C3 独立收益。

本轮另为量化评测入口增加 `--a1-reset-dir`，使冻结 `.npy` 直接替换初态读取；未使用旧专家示范 JSON 的 `success` 字段来伪造专家成功。该改动不改变模型、量化 profile、动作语义或 8 步 chunk。新入口代码已在 014 编译通过，SHA256 `b4efd426d521e0e44438df48b4e8aba39280d9e4c3160e0ac8ca1913d87207b6`。

## 归档范围和局限

服务器原始审计与生成文件位于 `/root/autodl-tmp/qvla-repro/artifacts/a1-014-preflight-20260930/`；小型结构化数据与 20 个候选状态在同名 `results/experiments/p2-shared-peft/20260930-014-a1-reset-preflight/`。完整归档为 `/root/autodl-tmp/qvla-repro/backups/20260930-014-a1-reset-preflight-final-033429-852d350/`，本机副本为 `backups/experiments/p2-shared-peft/20260930-014-a1-state-source-preflight/20260930-014-a1-reset-preflight-final-033429-852d350/`；两端清单 10/10 文件 SHA 核验通过，`session-results.tar.gz` SHA256 为 `5559bb07a6d016a933b549b5678fa3f5c02e56cea2883451089212eacac04681`。大模型与校准原件留在服务器持久盘，未放入普通 Git；具体见归档 `LARGE_FILES_NOT_IN_GIT.json`。**本轮无完成的 GPU episode、无 C0/C3 配对收益、无 A1 复现证据。**预检阻止了再一次环境错误下的长运行。
