# 035 本轮 B2/B3 预运行审计与阻断结论

**结论：B2 与 B3 均未运行训练或闭环评测，不能报告新成功率。** 用户开启的 035 克隆机经既有 SSH 密钥及隔离 host key 核验，hostname 为 `autodl-container-481a45991f-b03cd0b4`，GPU 为 RTX 4090 24 GB。检查时无训练/评测 GPU 进程。按新的持续工作约定，付费实例不等待代码开发或审阅；本轮保存审计证据后收尾关机。

## 已通过的物料核验

- B1（旧 LW）adapter SHA256：`a231d7fb3cbb239e648547d466a68e32922a228e540f0b83bd9394e278389bf8`；克隆文件与注册值一致。checkpoint `config.json` SHA256：`bcb688a66f3e94a42311b77d37b7e1484886701f7b2aa8e36d165dfccac92ae6`。
- W2/G128、W2/G64、W4 profile SHA256 分别为 `4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc`、`947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8`、`ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4`，与已登记物料一致。
- 新增只读审计器 `scripts/audit_peft_calibration80.py` 逐文件验证 80 个 Response-SVD 输入 SHA，并以源 `episode` 对照轨迹划分：80/80 均是互不重复的 `peft_train` 轨迹，未发现 `router_dev` 或 `offline_final_holdout` 混入。源清单 SHA 为 `ae493d1fe3e32a5e64fb3658fee19b3528fb1c65ecd14c7f92f14ecb48d13742`，轨迹划分 SHA 为 `d4903a2e9a1ee6b474f5215644ba68d494a8feb89354b3a2a88bbe13a9d48124`。这解除两张卡的 SVD 来源缺证阻断，但不等于全部训练与评测门禁通过。
- 按新机约定重新同步并核验 `backup_active_diagnostics.py` 与 `vla_shutdown_remote.py`，服务器持久盘 `closure-tools` 的对应 SHA 为 `698f0b210d4e0ecd5cecdea83122f6358fcf45468615b967010f0bda973a68a4` 和 `efed5fdf5e7a98c900306f9ceba823f4c2b417ae36194ec6ebe59fd3f17ff12c`。

## 阻断与下一步

1. **B2：** 现有训练入口只为语言 block 建立 LoRA 目标；评测器只接受一份语言 adapter，不能同时冻结 B1 并加载新视觉 adapter。视觉量化 `Linear` 白名单、精确目标 SHA、零输出、冻结性、保存重载和 10 步烟雾均尚未建立。旧 B1 相对 B0 的确认性任务聚类门槛也未过；按新卡只能作为探索性实验。
2. **B3：** 当前注册入口明确拒绝 B3；现有 LoRA loader 仅允许 A3 中的 20 个语言 W2 block，不能把 32 层全 W2 直接命名为 B3。A4 学生自身访问的 40 回合/80 观测、同观测 BF16 标签、全 W2 目标与 profile 的有效装载、零残差、10 步烟雾均未产生。不能用 A3 的 student-state80 或 B1 的 20 层 adapter 代替。
3. **下一轮在开机前完成代码：** 分别实现版本化 B2 与 B3 入口、严格 key/shape/目标验证、B2 冻结 B1 和 B3 新学生数据采集路径；在本地静态检查后再开机做各自 10 步 GPU 烟雾。通过后依各自 PRE-RUN CARD 做配对微测与最多 50 回合首片硬门禁。对复用开发 reset 的结果只报告开发证据；独立条件需另外通过真实评测顺序的可观测性门禁。

本轮未改变 backbone、训练数据、adapter 或 evaluator，也未访问 `offline_final_holdout`。历史开发 300 回合/配置为 A3=244、B0=260、B1=284、BF16=292、A0=294；它们不是本轮新测结果。已核验状态见同名 [结果文件](../../../../results/experiments/p2-shared-peft/20261005-035-b2-b3-preflight/preflight.json)。收尾归档、Git 提交与关机回执另记于本目录的 closure 文件。
