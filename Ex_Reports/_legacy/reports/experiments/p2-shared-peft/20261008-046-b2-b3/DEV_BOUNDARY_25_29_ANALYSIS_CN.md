# B2/B3 开发边界扩展：reset25–29

2026-10-08，046 实例。协议门禁为 `PASS_PROTOCOL_BOUNDARY_FIRST50`，七配置在相同的 50 个任务／初态键上严格配对，首个策略可见观测一致，动作及观测 trace、8 步 chunk、退出码与产物哈希通过审计。每配置先做的 reset25 微测与本片重叠，**不另计样本**。这些都是复用的官方开发初态，不能称独立复现或盲测。

| 模型 | 成功／50 |
|---|---:|
| A3（12L，无微调） | 42 |
| B1（12L＋语言扩展 LoRA） | 49 |
| B2（B1＋视觉 LoRA） | 49 |
| A4（全 W2，无微调） | 7 |
| B3（全 W2＋语言 LoRA） | 46 |
| BF16 | 46 |
| A0（全 W4） | 49 |

- B2 相对冻结 B1：1 rescue、1 break，净 0/50；任务聚类 bootstrap 95% 区间为 −6 至 +6 个百分点，exact McNemar p=1。视觉增量 LoRA 在此片没有净收益，但存在配对翻转，不能只看总成功数。
- B3 相对 A4：39 rescue、0 break，净 +39/50；十个任务均净改善，任务聚类区间 +62 至 +92 个百分点，exact McNemar p≈3.64×10⁻¹²。全 W2 语言 LoRA 在开发集上恢复幅度很大。
- B3 相对 B1：0 rescue、3 break；B3=46/50 仍低于 B1=49/50。不同底座且 B1 历史训练有 proprio 双重归一化风险，该差异只作描述，不据此推断 LoRA 结构或 routing 优劣。
- B2 与 B3：B2 相对 B3 有 4 rescue、1 break，净 +3/50；区间 −4 至 +18 个百分点，不能据此断言 B2 更优。

把不重叠的开发 reset20–24 与 25–29 合并，仅作开发边界描述：B1=97/100、B2=94/100，B2 相对 B1 合计 1 rescue／4 break；A4=17/100、B3=93/100，B3 相对 A4 合计 76 rescue／0 break。分片的成功率和损失模式有波动，因此继续已预注册、协议门禁驱动的 reset30–39 配对评测，不按成功率方向选择续跑。该续跑不训练新 adapter，也不接触封存留出集。

原始 25–29 门禁及每配置产物 SHA 见同名 `results/experiments/p2-shared-peft/20261008-046-b2-b3/boundary-25-29-v1-first50-protocol-gate.json`；服务器原件为 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261008-046-b2-b3-boundary-25-29-v1/`。30–39 的预注册卡和计划／锁文件也已分别保存在本报告目录和同名 results 目录。此前 B2/B3 的训练成本与首片诊断见本目录 `B3_FIRST50_ANALYSIS_CN.md`；此次仅增加评测成本，没有训练新模型。
