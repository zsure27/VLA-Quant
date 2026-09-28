# P2.5 配对闭环 headroom 检验：预注册卡

日期：2026-09-28。研究门禁：仅 P2.5；exact-12L backbone 与 blocks 18–19 不变。候选为已冻结的 data80/1000-step、rank8 Response-SVD Smooth-L1 Recovery-LoRA。训练覆盖与计算量同时改变的历史离线改善不作单变量因果解释。

## 问题与证据

先前 50 回合片段 C0=43/50、C2=43/50，没有可计算的 W4 headroom；C1=42/50。另一次 10 reset P2.5 学生访问状态的同状态 BF16 查询显示 LoRA 平均动作 MSE 与夹爪符号分歧改善，但唯一失败 episode 中 LoRA 的平均 MSE 也比未加 LoRA 的同状态探针低，说明离线改善不能推出闭环成功。历史 12L 411/412 与 14L 430/431 也存在 evaluator/source 口径漂移。

## 固定比较

- C0：exact-12L，无 adapter。
- C1：完全相同的 exact-12L，加冻结的 rank8 Recovery-LoRA；**C1 对 C0 的唯一模型变量为 adapter**。
- C2：同 evaluator 的 14L 静态参照，仅 blocks 18–19 升 W4，用于测量这个配对集合中的 headroom；不训练、不搜索 W4。
- 同一 evaluator 文件、profile、checkpoint、seed0、初始状态及每任务 50 回合；按 0–9、10–19、20–29、30–39、40–49 五片顺序，每片每配置 100 回合。所有结果仅属已使用过的历史/开发 states，offline_final_holdout 保持封存。
- C1 留存逐次动作和观测以定位失败；每片记录源代码、profile、adapter 哈希、原始 episode 结果和退出码。

## 事先定义的判读

主指标是 C1-C0 的配对成功差及 12L 失败而 14L 成功样本上的救回/新增失败；另报逐任务差、episode bootstrap 区间、夹爪与动作分歧、参数/字节/时间。只有 C2-C0>0 才计算 W4 恢复比例，否则记 `null`。五片未全部完成时只报告已完成分片的开发结果，不混作 Spatial500。若闭环收益不稳定或失败来自学生访问分布，继续 P2.5 的同状态 BF16 重标与训练控制；不进入 P4/P5/Router。若额度到达收尾门限，则按约定停止后续分片、归档并关闭 107。
