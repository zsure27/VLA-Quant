# 059 语言 W2 全层 rank-8 Recovery-LoRA：PRE-RUN CARD

状态：**预注册中；尚未启动 GPU 训练/策略评测。** 机器已验证为用户开启的 059 克隆，RTX 4090 空闲。结果归类为复用官方初态的开发压力测试；不声称独立 seed/reset、新初态泛化或 holdout。

## 问题、变量与结论边界

**问题：** 在 exact-12L 混精 backbone 与 C3 完全相同的训练数据、教师标签、优化器和 rank 下，把 Recovery-LoRA 从 blocks 18–19 扩展到全部语言 W2 blocks，能否相对同条件 C0/C3 增加闭环净成功并控制 rescue/break？

**唯一处理变量：** adapter 目标范围由 blocks 18–19 扩展到语言 W2 blocks `{0–7,16–19,24–31}` 的全部合格 Linear（self-attention Q/K/V/O 与 MLP gate/up/down），每层 rank 8。量化 profile、backbone、数据、同观测 BF16 教师标签、Response-SVD 初始化、Smooth-L1、训练步数和评测都冻结。范围变宽必然增加可训练参数，因此该比较估计的是“覆盖范围 + 随之增加的适配容量”的工程总效果；**不能单独归因于层数/覆盖率**。不得临时调 rank、学习率、步数或目标层。

**评测条件：** LIBERO Spatial 官方 reset 数组 20–49，model seed 0、`env_seed=1`、`seed_protocol=paired`、真实 8 步动作 chunk。每个条件使用相同 task/reset/初态哈希。该条件与本仓库已有开发数据共享官方初态数组，所有结果标 `DEVELOPMENT_STRESS_REUSED_INITIAL_STATES`；不得称独立复现/新初态盲测，不触碰 `offline_final_holdout`。同片包括 12L/C0、冻结 C3、宽语言 LoRA、BF16、完整 AWQ W4。历史 107/035 成功数只作背景，不能代替本片对照或填恢复率。

## 冻结物料与实机哈希

主机：`autodl-container-5e9a44960d-2363ecb7`，SSH 端口 21271；RTX 4090 24 GB。代码在隔离副本 `/root/autodl-tmp/VLA-Quant-lora-expand-20260930`，原克隆 `/root/autodl-tmp/VLA-Quant-p2c-20260925` 保持不动。

| 物料 | SHA-256 |
|---|---|
| checkpoint config.json | `bcb688a66f3e94a42311b77be28eafc742d66605d339f8d9acfedb8da4703ea6` |
| checkpoint model.safetensors.index.json | `ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659` |
| checkpoint shard 1 | `2809bd7be9422315c5ecbe91eea612f5b02925f37e0051728ad45bc993c79251` |
| checkpoint shard 2 | `a00a7c5f2b6586ccfc89c693a9c36f3552ff455ff2b1bfea3e92642e4cd2b6d3` |
| checkpoint shard 3 | `a894b7230a08b471af55c57dd7385fd3b51fae2c4d9307a36a8017ef57abf22a` |
| checkpoint shard 4 | `a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60` |
| base W2 profile | `4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc` |
| full AWQ W4 profile | `ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4` |
| primary vision G64 W2 profile | `947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8` |
| exact-12L C3 rank8 adapter | `67cd6a6d4ec75d9173ce95b5f8f21c99b6562bfea58ea80947aa5640a335710a` |
| student-state80 manifest | `858d844ed55bf3d56ba6c95d40be8480659d69dabe03ecda703ff0918856d642` |
| peft_train Response-SVD calibration manifest | `ae493d1fe3e32a5e64fb3658fee19b3528fb1c65ecd14c7f92f14ecb48d13742` |
| frozen BF16 teacher manifest | `d58aa6315ba0aae40cbde0ea17dfbf33ccdd6d89cf727e460157e5dc39bf2efe` |
| candidate loader `run_eval_official_quant.py` | `aaf92ce5257d120437ef9830a0ed57341f8714a83755dfdabebd77c1a38991cf` |
| candidate loader/guard `diagnostics/probe.py` | `bea54c391ad4ec8dd1a88adb1fa04e0a065a1b7996128cd9186f98ee93fdfb26` |
| connected target list | `513e1d937d1b345ec00fdb19eee73709b8704941b18c21e2258bdf795f495b3a` |
| frozen official rollout evaluator | `run_libero_eval.py` `fe37e8097c286e1946e5194a1f817a2b2362d167a6a5a49443c5aa20314d8873` |
| interventions source | `awq_interventions.py` `1f728b02028c915bf18dc6c1328d3e7e4a47c24c30c04bf151539b4cab095b41` |
| quantization adapter source | `official_quant_adapter.py` `6179991b5d92ef4d8f2db956dee9b06851778c02569d5e3299c82843f0ae8171` |
| action preparation source | `action_jacobian_batch.py` `9e182bcfc6616221360d6d8c9216ddcafc797125e9aa3df141f5dfcdf6a5c886` |
| preregistered experiment launcher | `run_20260930_059_language_w2_all.sh` SHA `f409988e409c78fcc11b0899abf441956bb8fae4f6308083dd8fcee5a7ae380e` |
| training smoke verifier | `verify_20260930_059_lora_training_smoke.py` SHA `933135923a2df065b639b7a57b85240f45b8096c6eb44c8f2e3463b78b46dc88` |

配置来源：仓库 `configs/backbones/awq_w2a16_12l_mixed_spatial_v1.json`；视觉 DINO W2/G64、SigLIP W2/G128；语言 blocks 8–15 与 20–23 为 W4，其余 W2；保护模块保持高精度。`w4.pt` profile 覆盖 422 个线性模块（视觉198、语言224），作为完整 W4 条件参照。BF16 使用已验证的 same-loader 命令 `--weight-bits 4 --profile w4.pt --awq-scope none`，因此并不应用 W4 量化；W4 对照使用同一 profile 且 `--awq-scope all`。训练样本为现有 80 个 student-visited 训练 reset0–3 观测；与评测 reset20–49 不重叠。Response-SVD 校准只来自 `peft_train` 80。不得重做或混入其他数据。

## 训练协议与成本记录

- 冻结初始化：所有 20 个目标 block 都按现有实现从相同 `peft_train` Response-SVD 校准输入初始化；这**不是**把 C3 adapter 扩展/续训。C3 仅作闭环比较锚点。逐层记录初始化算法、Response-SVD 残差覆盖和参数量；若 20 层无法全部按同一流程初始化，停止。
- 固定：student-state80 同观测 BF16 标签、80 样本、1000 optimizer steps、Smooth-L1 beta=0.1、学习率 `1e-4`、AdamW、seed7、8×7 action、同一 batch 语义。此次不改变损失/优化量。
- 先做 10-step GPU smoke：实际目标路径必须恰为 20 blocks × 7 Linear 类型；记录参数数、adapter 字节、损失/梯度有限、冻结量化主干/profile 哈希不变、保存重载后输出一致、峰值显存/耗时。10-step 结果不作训练效果证据；只有烟雾通过才运行 1000 步。
- 量化 profile 是 fake-quant 模型状态；adapter 字节/参数不能被描述成 packed W2 部署压缩或推理加速。记录 8-step chunk 延迟、模型常驻显存与加载开销。

## 烟雾、分片、统计与停走门槛

1. EGL 显式 `MUJOCO_EGL_DEVICE_ID=0`；真实 rollout 源码在 paired 协议下先 `env.seed(environment_seed)`，紧接着 `seed_all(model_seed)`，然后写入同一官方 `initial_state` 并按 8 步 chunk 执行。用实际评测入口与渲染、预处理检查同条件重复的两路相机/proprio 输入哈希与 reset 注入；检查旧 seed A1 的无效性不被误当本轮独立性。条件烟雾只证明执行/输入记录有效，不主张环境独立。
2. 训练与 adapter save/reload smoke 通过后，先对 offsets20、每任务1回合做 10 对/配置的政策微型闭环，核对配对键、成功定义、8步 chunk trace 与评测命令/所有哈希；pilot 不并入主分析样本。
3. 首个正式片严格为 offsets20–24，即 50回合/配置 × C0/C3/宽 LoRA/BF16/完整W4。片后硬停止，复核命令、profile/checkpoint/adapter SHA、逐回合 manifest、异常、配对完整性与真实 trace；不按正负结果改变余下样本量。通过协议门禁后才解锁已预注册 offsets25–49（再250回合/配置）；若 smoke/协议失败即停止扩展、保留产物、分析后收尾。
4. 样本量固定为 offsets20–49，共300回合/配置，五种策略条件同片配对。逐配置及比较对象报告 `S_A/S_B`、双方成功/失败、rescue、break、net、按 task 聚类的配对 bootstrap 95% CI、精确 McNemar、每任务 net/worst task、reset/初态键和缺失/异常。主增量比较宽 LoRA–C3 与宽 LoRA–C0；不把 rescue+break 本身称作可路由冲突。
5. 对 BF16、完整 W4 报告同片成功率与配对四格。仅当同片 `S_ref−S_C0>0` 才给 `recovered_fraction=(S_A−S_C0)/(S_ref−S_C0)`，否则 `null`。若 W4/BF16 的模型加载或 profile/hash 不一致，仍保存其他结果，但该参照标不可比，不填历史数字。
6. **进入视觉扩展 LV 的预注册规则：** 宽 LoRA 相对 C0 的净改善至少 +5 个百分点且 task-cluster 95% CI 下界大于0；相对冻结 C3 的净改善至少 +3 个百分点且其配对 95% CI 下界大于0；任一任务不得比 C3 多损失超过2个成功/30回合；rescue/break 均须公开；训练成本/推理延迟无未解释的数量级异常。300回合不能达到门槛时不启动 LV，只做错误分析。该门槛是工程推进门，不宣称统计等价或独立泛化。
7. 新训练/评测总时间若不能在实时额度剩余15%前完成，先完成可审计 smoke/首片并在门限收尾，不启动无法完成的长片。仅用 Codex 当前五小时额度；15%归档、备份、同名 `reports/results`、固定 `zsure27/VLA-Quant` 推送/SHA核验；10%停新 GPU 并关当前059，留至少3%。

## 输出与审计路径

- 唯一运行 ID：`20260930-059-language-w2-all`。
- 服务器运行/持久盘：`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260930-059-language-w2-all/`；控制器：`/root/autodl-tmp/qvla-repro/control/p2-language-w2-all-059-20260930/`。
- 本地忽略副本：`backups/experiments/p2-shared-peft/20260930-059-language-w2-all/`。
- 报告与小型配对数据：`reports/experiments/p2-shared-peft/20260930-059-language-w2-all/` 与 `results/experiments/p2-shared-peft/20260930-059-language-w2-all/`。
- 机器专属 PRE-RUN 卡不得替代执行中的逐命令、哈希、实际路径与偏差记录；完成后另写结果分析/停止门禁。
