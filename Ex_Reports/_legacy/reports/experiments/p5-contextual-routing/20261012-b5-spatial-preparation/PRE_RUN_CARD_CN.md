# B5-pilot-v1 PRE-RUN CARD（2026-10-12）

STATUS: CODE_AND_CPU_CONTRACTS_PASS_REAL_INPUT_AND_GPU_GATES_PENDING
USER_AUTHORIZATION: 先解决QVLA/AWQ基线问题，再探索Spatial上的B5，并比较BF16/A0/B4/B5。
TYPE: DIAGNOSTIC；同任务扩大开发评测，非独立/盲测。
ADMISSION: 不因论文目标成功率放行；先通过实际基线、输入、量化和因果契约。

## 假设与变量

HYPOTHESIS: 同一A4底座的B3/B4存在可由当前观测预测的局部互补，Top-1逐chunk选择可改善相对B4的rescue/break。
VARIABLE: 在冻结的两候选adapter之间使用当前因果状态选择；不修改量化基底或专家参数。
CONTROL: 主表BF16/A0/B4/B5；内部必须保留B3与固定专家控制，才可解释超过最佳静态策略。
PRIMARY_METRIC: 事前指定B5−B4与B5−B3两项配对闭环净改善、rescue/break与任务分层95%区间；只有两项均显示稳定正收益才主张超过最佳固定。不得看完450结果后只选择有利参照检验。
SECONDARY: rescue/break、每任务成功、夹爪首次分歧、选择/切换次数、bank参数/字节、峰值VRAM、batch1 chunk延迟。
CLAIM_LIMIT: 两候选覆盖/容量不同；本pilot不证明同预算新专家或跨任务泛化。

## 模型与数据

BASE: 经本轮基线修正后固定的A4全合格W2；确切checkpoint/profile/targets/runtime SHA由实机回执填充，未填不放行。
E0: 冻结B3语言adapter，视觉分支为零。
E1: 冻结B4语言＋视觉adapter。
ROUTER: 独立小型因果编码/分类器，在第一可选视觉层前选择。已固定402→64→2、25,922参数、AdamW 500步；同优化预算instruction-only对照另跑450，random首50；完整正式预算2,750。
FORBIDDEN: 当前post-adapter H17、推理阶段任何当前专家后的特征、未来成功、GT、推理时教师查询。
TRAIN: reset0–3完整训练轨迹中的分层观测；两专家同观测回放与冻结BF16动作teacher形成监督。新增清单/覆盖/teacher查询上限须实现时固定。
ROUTER_DEV: reset4，保守排除于主评测；训练/校验按整个回合隔离，架构/阈值选择的有限候选与停止规则先固定；不读5–49历史成败来训练/调规则。
EVAL: 10 Spatial任务×reset5–49=450/模型，正式四配置1,800回合；完整B3内部控制则2,250。task-only/random/shuffled控制另计，完整资源预算不能只按上述主配置数量估算。
HOLDOUT: offline_final_holdout不读标签、不改角色；PTQ轨迹来源不能靠reset编号排除。

## 运行前必须补齐的契约

1. 基线审计及新实际模型/协议门禁通过；W2有效权重、保护范围、checkpoint或动作语义改变时，须对应的新版本训练和验证，仅核验旧adapter的SHA身份不足以建立兼容。
2. 可运行B5注册/训练/切换入口，新checkout、输入哈希和不可变顺序计划。
3. 固定E0/E1逐输出复现B3/B4；base和expert冻结；router参数白名单。
4. 无残留gate/cache状态、保存重载一致、当前特征在专家选择前。
5. 每8步chunk仅一次选择；不在chunk中更换专家；没有测试标签或未来输入。
6. 10步GPU烟雾、配对微测、首50硬审计；完整manifest/初观测/动作trace/退出码。
7. 数据初态与观测交集、task指令映射、PTQ来源暴露审计；5–19不可沿用旧20–39回执。
8. 磁盘/时间/实时额度足够；准备服务器持久盘、本机完整副本、报告/小结果和SHA。

## 固定执行与停止

首片reset5–9各50；通过硬协议审计后接10–19/20–29/30–39/40–49，各100。
微测与正式重叠不增加样本。protocol通过后不按成功率方向挑选继续。
若基线、因果、数据或实现失败：保留原始数据；版本化修复；无法修复或需审阅即报告、备份关当前实例，避免空跑。
最终报告必须将探索成功、负结果和未完成项区分；B5胜过B4但未胜过最佳固定不能证明路由必要。

## 本机阶段结论

代码源hash与旧A0指标重算通过；未完成新GPU数值门禁、AWQ论文严格配方复现、B5实现或B5实测。


## 额度刷新前实现进度

新增bank、8步状态机、402维因果特征与CPU契约；实际6项pytest通过（6.65s），含FP32/BF16 toy固定专家严格等价、切换、冻结SHA、重载、重复hook拒绝和状态归一化。未完成真实OFT/GPU、外围注册/训练/运行/计划。待额度实时刷新继续，不能开卡长跑。

## 最终CPU审阅

详见CPU_REVIEW_CN.md。B5当前8项CPU测试通过，真实环境门禁因2GiB实例被系统Killed仍未通过；开卡先真实输入桥接、BF16/A0配对微测和首50、query10 bank/static等价，再进入router采集训练与450开发评测。当前只完成CPU可运行代码与前置合同，未产生正式B5参数或成功率。
