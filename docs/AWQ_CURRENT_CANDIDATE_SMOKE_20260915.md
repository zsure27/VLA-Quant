# AWQ 当前 W2 候选闭环冒烟记录（2026-09-15）

## 候选合同

- 量化目标共 422 个：语言 224、DINO 主视觉 93、SigLIP 融合视觉 105。
- 语言侧使用 W2/G128，并移除 160 个额外 clipping；q/k 原本无 clip。
- DINO 主视觉使用独立校准的 W2/G64；SigLIP 融合视觉保留 W2/G128；视觉 clipping 保留。
- 实际运行日志确认 `422/422`，分组计数 `G128=329, G64=93`，`removed_language_clips=160`。
- 合同产物：`/root/autodl-tmp/qvla-repro/artifacts/awq-current-candidate-contract-20260915.json`。

## 本次执行结果

首次 10 回合命令在模型加载和 422 层量化完成后，于创建首个 LIBERO 环境时终止，未进入 rollout，因此没有成功率结果。失败日志保存在：

`/root/autodl-tmp/qvla-repro/eval/awq-current-smoke10-20260915-103102/console.log`

根因是运行环境漂移：`robosuite 1.4.1`、`mujoco 3.13.0`，而 LIBERO 项目声明 `robosuite==1.4.0`，并基于 MuJoCo 3.1 系列接口。3.13.0 导致关节枚举比较和质量矩阵接口不兼容。环境已恢复为：

- Python 3.10.14
- NumPy 1.26.4
- robosuite 1.4.0
- mujoco 3.1.1

恢复后，以未加载 VLA 模型的原生任务 0 环境检查通过：

`ENV_OK pick up the black bowl between the plate and the ramekin and place it on the plate 18`

## 验证

- `python -m py_compile qvla/run_eval_official_quant.py`：通过。
- `python -m unittest discover -s tests -p test_awq_interventions.py`：9 项通过。
- 当前候选真实 profile 合同检查：PASS。

## 下一步

额度在环境修复后已进入收尾阈值，本轮没有重新启动 rollout。下次从相同命令先跑 AWQ 候选 10 回合管线验证；通过后增加相同 task/episode/env seed 的 BF16 对照，随后再扩为每任务 20 回合。离线动作 MSE 只用于筛选，正式判断以配对闭环成功率为准。
