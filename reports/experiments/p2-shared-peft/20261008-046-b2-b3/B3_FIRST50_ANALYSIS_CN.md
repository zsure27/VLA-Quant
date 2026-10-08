# 046 B3 全 W2 语言 Recovery-LoRA 首片分析

执行代码锁定 Git `85a0934f4b604d409b6a0b64b209a3f0ec4a88a2`。不可变计划 `20261008-046-b2-b3-proprio-v4-B3` 的 16 个阶段全部退出 0，终态 `AWAITING_GATE_REVIEW`；微测与首片硬审计分别为 `PASS_PROTOCOL_MICRO` 和 `PASS_PROTOCOL_FIRST50`。本报告只使用首片 50 个严格配对回合；微测中的 reset20 与首片重叠，不增加样本数。

## 模型、数据与契约

- A4：全 W2 未微调底座；B3：同一 A4 加覆盖全部 32 个语言 W2 block 的 rank8 Recovery-LoRA。B1 是 12L 混精底座上的历史扩展语言 LoRA，BF16 与 A0 全 W4 是精度参照。B2 是 B1 上另增视觉 LoRA，属于不同底座和干预，不与 B3 当作单变量直接比较。
- B3 训练只从 A4 在十任务、训练 reset0–3 的 40 条学生回合中取前两次策略查询，共 80 个观测；冻结 BF16 教师在相同观测重标。`student-state80/manifest.json` 明确标记 `policy_normalized_proprio`，避免把策略已归一化的 proprio 再归一化。Response-SVD 的原始 `peft_train` 输入仍只归一化一次。
- 10 步烟雾、零残差/重载等价、冻结底座哈希和 1000 步训练契约均通过。B3 有 19,988,480 个训练参数，adapter 序列化 40,133,618 字节，1000 步训练约 362 秒。SVD 校准 spool 预估 10,202,644,480 字节，开跑前实测可用 19,439,603,712 字节，预留 6,442,450,944 字节。
- 评测为 LIBERO Spatial 十任务、复用官方开发 reset20–24，每配置 50 回合；不是新初态盲测，也不是独立 A1 复现。五配置的命令、源与产物哈希、逐回合 manifest、初始观测、有限 7D 动作和 8 步 chunk 语义由硬门禁审计。

## 严格配对闭环结果

| 配置 | 成功/50 | 相对 B3 的解释 |
| --- | ---: | --- |
| A4：全 W2 未微调 | 10 | B3 的同底座主对照 |
| B3：A4 + 全语言 rank8 LoRA | **47** | 本轮候选 |
| B1：12L + 历史语言 LoRA | 48 | 不同底座、旧训练状态空间有风险 |
| BF16 | 49 | 高精度参照 |
| A0：全 W4 | 47 | 量化参照 |

B3 相对 A4 为 **37 rescue、0 break，净增 37/50（74 个百分点）**；改善分布在 9/10 个任务，任务聚类 bootstrap 95% 区间为 [52, 92] 个百分点，精确 McNemar p=1.46×10⁻¹¹。这个配对信号说明在当前开发 slice 上，全语言 LoRA 能大幅恢复全 W2 的闭环能力。

B3 相对 B1 为 **1 rescue、2 break、净少 1/50**，任务聚类区间 [−8, 4] 个百分点。相对 BF16 为 0 rescue、2 break；相对全 W4 为 2 rescue、2 break，净差 0。B3 的三次失败在任务4/reset21、任务5/reset20、任务5/reset22；其中任务5/reset22 连 BF16 与 W4 都失败，不能归为 B3 独有量化错误。任务3/reset24 则是 B3 成功而 B1 失败的一个互补候选，单个开发回合不足以证明可泛化的路由价值。

## 与 B2 的关系及研究判断

同轮 B2 首片为 45/50，相对冻结 B1 的 48/50 有 0 rescue、3 break；其视觉增量没有观察到正收益。B3 使用更难的全 W2 底座且拥有约 20.0M 参数，其 47/50 不能解释成“视觉与语言哪个更好”的受控消融。B3 证明了本 slice 上静态共享语言 LoRA 的强恢复潜力，但与 B1/BF16 的 1–2 回合差距处于高成功率天花板区，且 B1 历史训练有双重 proprio 归一化风险。

目前没有足够证据启动 Router：B3 对 A4 的 37 次 rescue 没有 break，不显示需要按上下文规避的冲突；B3 对 B1 的 1/2 互补翻转混入了底座、adapter 来源和训练分布差异。也不能把开发 reset20–24 的 47/50 推广成全 W2 已恢复到 W4/BF16 的总体性能。

## 门禁结论与下一步

本轮两个预注册 B2/B3 计划已经完成。长评测扩展仍为 `LOCKED`：若要确认 B3 的稳定收益，先按 `docs/EVALUATION_SMOKE_GATE_20260929_CN.md` 寻找真正可观测、可复现且无训练/开发重叠的新评测条件，先无模型真实调用顺序检查，再配对微测和至多 50 回合首片；不能仅改 `env_seed` 或重复同一开发 reset 当独立样本。若没有该条件，保持 B3 为开发证据，不继续消耗 046 GPU；保留 A4/B3/B1/BF16/A0 的原始轨迹与失败样本供后续归因。Router、P4/P5 和 offline_final_holdout 仍锁定。

原始小型门禁、训练 artifact、训练样本 manifest 和 SVD 空间预算见同名 `results/experiments/p2-shared-peft/20261008-046-b2-b3/`；服务器原始目录为 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261008-046-b2-b3-proprio-v4/`。最终双份归档路径、SHA、GitHub 提交和 046 关机回执记录于本目录的收尾文件。
