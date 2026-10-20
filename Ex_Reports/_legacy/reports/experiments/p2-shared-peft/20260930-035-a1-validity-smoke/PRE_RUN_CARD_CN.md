# 035 A1 reset 条件有效性烟雾卡

## 冻结问题与边界

- 唯一问题：前瞻性生成的每任务 `index 0` reset，在真实冻结 OpenVLA-OFT BF16 策略下是否表现为可执行的 Spatial 任务。
- 本阶段只做 10 个 BF16 回合（每个 Spatial 任务一个 reset），不训练 adapter、不运行 C0/C3、不估计 PEFT 收益、不宣称 A1 独立复现。
- 唯一变化是初始状态数组：比较对象为已使用的官方状态与每任务新生成的 `index 0`。动作、任务、模型、观察预处理、评测顺序、8 步 chunk、环境 seed 与 model seed 固定。
- 不因单个失败替换 reset 或挑选对 C3 有利的状态。此 pilot 通过只表示可以继续检查，不表示 20 个 reset 全部有效、与所有示范完全无重叠，或支持独立泛化。

## 035 实例与运行物料

- 实例：`autodl-container-481a45991f-b03cd0b4`，SSH 端口 `15924`；RTX 4090（24564 MiB），GPU 检查时空闲。
- 克隆仓库 HEAD：`852d35052526f79229362ac43571a009db0167e7`。克隆工作树已有本地改动及未跟踪文件；运行不从该工作树导入评测器，也不覆盖它们。
- 实际评测入口：`/root/autodl-tmp/VLA-Quant-p2c-20260925/qvla/run_eval_official_quant.py`，SHA256 `b4efd426d521e0e44438df48b4e8aba39280d9e4c3160e0ac8ca1913d87207b6`。
- OFT helper：`/root/autodl-tmp/qvla-repro/overlays/awq-p0-stage-20260923/oft/experiments/robot/libero/run_libero_eval.py`，SHA256 `fe37e8097c286e1946e5194a1f817a2b2362d167a6a5a49443c5aa20314d8873`。加载该 overlay 的 `PYTHONPATH`，并在开始前断言导入路径与 `initial_state_offset` 参数存在。
- 初态目录：`/root/autodl-tmp/qvla-repro/artifacts/a1-014-preflight-20260930/fresh-reset-pilot/`。10 个 `task-00.npy` 至 `task-09.npy` 在本机与035逐文件 SHA256 完全一致；完整 SHA 清单随原始结果归档。
- BF16 checkpoint：`/root/autodl-tmp/qvla-repro/models/openvla-7b-oft-finetuned-libero-spatial/`；4 个模型分片和 OFT adapter 文件存在。加载器在运行时执行 checkpoint/profile identity 检查，运行前后保存物料清单。
- 同 loader 的 BF16 参数：`--method awq --weight-bits 2 --activation-bits 16 --awq-scope none`；固定 W2 profile 仅为 evaluator 接口必需输入，`none` 路径不得应用 scale、clip 或权重量化。
- 输出：服务器 `/root/autodl-tmp/qvla-repro/eval/a1-035-fresh-reset-bf16-validity-20260930/`；本机 `results/experiments/p2-shared-peft/20260930-035-a1-validity-smoke/`。

## 硬停止规则

任一代码、helper、profile、reset SHA 不符；overlay 导入断言失败；模型输入/任务状态未记录；评测命令未生效；输出回合数与10不符；或评测退出异常，立即停止并标成技术失败。不得将无 episode 的启动错误写成模型失败/成功。若有任务失败，只调查失败样本和评测语义，不更换状态后重跑。

即使 10/10 成功，后续也只能进入已登记的 C0/C3 配对策略微型 smoke（每任务同一 `index 0`，不超过10对），先重新冻结其卡片；不得自动进入50回合首片或长 A1。训练示范 reset 元数据缺失仍是独立性审计硬限制；在解决前，任何新 reset 结果最多称“前瞻性 reset 条件的开发性诊断”。
