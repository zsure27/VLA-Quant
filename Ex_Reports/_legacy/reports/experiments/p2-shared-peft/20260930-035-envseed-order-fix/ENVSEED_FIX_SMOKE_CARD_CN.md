# 035 环境 seed 顺序修复烟雾卡

状态：v1 尚未执行，已在运行前由 v2 取代。v1 只记录 `set_init_state()` 之后的策略输入，无法区分“环境 reset 没变化”和“变化被固定初态覆盖”；保留其内容作审计，不作为执行卡。执行卡为同日期目录下 v2 版本。

## 问题与唯一变量

原配对评测在每个 episode 先调用 `env.seed(environment_seed)`，随后 `seed_all(model_seed)`。LIBERO 的 `ControlEnv.seed` 只重设进程级 NumPy 随机流，而 `seed_all` 又重设 NumPy，因此 reset 实际使用模型随机流。修复后顺序为 `seed_all(model_seed) -> env.seed(derived_environment_seed) -> env.reset() -> set_init_state()`。

唯一比较变量为 `env_seed=0` 对 `env_seed=1`。固定同一官方 LIBERO Spatial task、每任务 reset index 5、model seed 0、初态数组 SHA、10 个稳定等待步、固定 no-op 动作及后续 8 步、OFT/LIBERO/evaluator/preprocessor/checkpoint/processor、动作与图像流程。每个 task/seed 从新环境重复两次。这里的 reset 是历史开发初态，结果不能称盲测或独立 holdout。

## 观测与记录

无模型地沿用评测器真实调用顺序。每个 task/seed 保存 query0 与固定 8 步后的：两路原始图像、robot state、sim qpos；评测器同一 `prepare_observation` 的 resized 双图与七维 proprio；OpenVLA center-crop + processor 后的双视角 `pixel_values`、`input_ids`、`attention_mask`；按冻结 checkpoint 的 proprio stats 归一化后的状态。processor 的 BF16 像素张量转成数值完全可表示的 FP32 后记录 SHA 和差异。逐项保存哈希、shape、dtype、跨 seed 最大/平均差异与变化比例，以及同 seed 重复差异；另记录 reset 预先成功、等待步 done 和固定步 done 标志。

## 事前门槛

- 四十次环境初始化和 80 个采样点都完整完成；每个 task 的同 seed 两次 reset 内容和 model-input 重复差异必须为零，或低于事前 noise floor。no-op 阶段不得提前结束/报错，初始 state 不得已满足任务成功条件。
- 跨 seed 的 model-input 差异必须超过同 seed 重复的最大噪声；query0 的实际 `pixel_values` 或归一化 proprio 至少在 8/10 task 改变，且后 8 步观察仍有可复现差异。只看 `env_seed` 字段、原始像素或脚本自报顺序不算通过。
- 若 10/10 固定状态下仍无策略可见差异，不运行 C0/C3 env-seed 评测；先单独诊断 `set_init_state` 是否覆盖环境随机化，并为任何替代 reset 生成流程另立单变量烟雾卡。
- BF16 10/10 不是门槛。其已完成的 9/10 与 task 4 失败完整保留；不得据此过滤 task 4 或选择 reset。

## 通过后的下一步

仅当上面的 seed smoke 通过，才冻结另一张 C0/C3 paired micro 卡：沿用这 10 个 task/reset，固定修复后的 evaluator、环境随机种子、model seed、exact-12L mixed profile 与 block18-19 C3 adapter；跑 C0/C3 各 10 个 episode。报告 `C0`、`C3`、`rescue=C0失败且C3成功`、`break=C0成功且C3失败`、净差和每任务结果。该 pilot 只检物料、配对、trace 与 seed 条件，不据正负翻转挑任务、替换 reset 或决定长测样本量。reset/训练示范重叠仍未证明前，不声称独立泛化。
