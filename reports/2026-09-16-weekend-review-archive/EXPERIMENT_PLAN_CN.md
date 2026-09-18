# 后续实验：分阶段安排、判断标准与资源控制

制定日期：2026-09-16。分析阶段无 GPU；用户随后提供从 014 克隆的 **107 机，SSH 端口 31263**，授权分析完成后实验。107 的数据新旧以连入后的文件/commit/hash 核对为准；不能因叫“克隆”就跳过核验。

## 1. 为什么按这个顺序

当前 W4 已过 50 状态 Spatial 开发门槛，语言 W2 是优先瓶颈，SQ 仅平滑控制尚未过关。下一步同时推进两个有限任务：**W2 语言损伤定位**与 **SQ 数值控制**；恢复训练只接在对应控制通过之后。视觉缩组已有离线收益，先固定为备选，避免同时改语言、视觉、训练和路由而无法归因。

### 本周末交流前的最小交付

| 优先级 | 交付 | 当前状态 / 下一步 |
|---|---|---|
| 必须 | 三条基线状态、真实数据边界、已有诊断图、精读与方法定位 | 本报告已整理；当前结果不是新 GPU 实验 |
| 必须 | W4 配对 50 状态成功矩阵和版本证据 | 已完成；补未测的动作/效率项仍待实验 |
| 高 | 语言 W2 四个 8-block 阶段的坐标一致 W4 rescue | A 脚本本地已准备，先跑 0–7、8–15，再决定 16–23、24–31；此前未运行 |
| 高 | SQ T0/T1 及首异常层定位 | 先复取旧 controls，再新生成 32 帧控制与逐层记录 |
| 条件 | 静态 Scale-PEFT/Recovery LoRA 小规模可行性 | 需梯度/保存/整数码契约、训练数据检查通过；不能保证周末前完成 |
| 后续 | 专家互补、Router、四 suite/full benchmark、真实 kernels | 不挤掉前述可信控制；有证据再晋级 |

有 GPU 时先运行最小有信息增益测试，模型加载后形成有限队列；每两次短实验或每一次长实验检查额度。不能把“GPU利用率暂时0”独立当成故障：数据加载、CPU仿真、保存是有效工作阶段。应同时检查进程、日志更新时间、CUDA状态和预期阶段。

## 2. 统一实验协议

1. 固定 checkpoint、运行代码 commit/dirty patch、官方仓库版本、attention mask/backend、422 targets、归一化/夹爪转换、8-step chunk 和种子。对变体保存 profile 的 SHA256、scale/clip/zero-point/group 信息与实际应用目标。
2. 固定 train/calibration/dev/final。轨迹级隔离，不将同轨迹不同帧分到不同集合。每轨迹按开始/接近/接触/释放多帧采样要记录规则和时间位置；开发阶段选型后封存 final。现有校准/诊断/验证都是开发资产。
3. 闭环按任务和初态严格配对，保存官方初态 index/hash、model/env seed。当前 init0–4已查看，之后的 pilot/final index需先确认库数量并预先冻结，例如 pilot5–9、final20–49；只是待核验划分建议。完整500次评测可以含既见状态，但必须分开报告开发重复与未见结果。
4. 主要终点是任务成功率及配对差；连续动作分位置/旋转、夹爪决策与阈值附近margin，teacher action L1/MSE/cosine是机制指标。GT action L1 仅在真实示范动作与坐标/归一化核对后报告。不可将不同 rollout 的时刻 t 直接相减称 teacher error。
5. 小样本仅筛选；固定任务分层重采样区间与 exact配对检验报告其限制。多次复跑同初态不增加独立环境状态数；多随机种子作为重复测量，另报算法方差。不得依结果好看提前停样本或隐藏失败候选。
6. fake quant 与 real packed 分表：理论平均位宽含排除模块、scale/zero-point、group padding、adapter和router；真实尺寸以部署文件和驻留tensor计。延迟需同步CUDA、预热、固定batch/图像数/chunk、P50/P95；另列加载与切换开销。不要将论文速度复制进本项目。

## 3. 实验矩阵及判据

| ID | 问题与变量 | 必须的对照 | 数据 / 图表 | 晋级与停止条件 |
|---|---|---|---|---|
| G0 | 新实例是否继承最新数据与可运行环境 | 014备份commit/profile/manifest | inventory JSON、GPU/环境/代码hash、启动日志 | 无GPU或数据/环境不可修复：保存故障信息并关机，不启动大下载来空耗 |
| B1 | W4是否稳定且动作数值正确 | 同入口BF16，固定当前W4profile | 50状态现有矩阵；新增同观测误差/夹爪/速度/显存 | 出现目标范围、初态或mask偏差先修；新样本明显退化则暂停W2训练 |
| Q0 | SQ仅平滑是否数值等价 | T0 BF16重复、T1 smooth BF16；局部FP32 | 每层绝对/相对误差、action-token cosine、夹爪margin、首异常层图 | T1未过经重复噪声校准的门槛：只定位，不做A4恢复训练 |
| Q1 | SQ的W/A各自影响及交互 | T0/T1/W4A16/W16A4/W4A8/W4A4；alpha固定 | 多精度离线表、误差CDF、token饱和率、同初态闭环 | Q0通过后开展；不能总MSE相减作贡献率 |
| Q2 | A4困难是否来自token类型/范围 | 原per-token absmax、相对裁剪ρ、token-type尺度、少量A8 rescue | visual/instruction/proprio/action-token范围、q码直方图、饱和率、闭环 | 稳定改善再用STE训练；参数扰动不改quant结果则舍弃无效scale |
| L0 | 哪个语言阶段对端到端最敏感 | language-only W2无clip，0–7/8–15/16–23/24–31 W4 rescue | 32开发帧逐样本配对差、gripper、全前向层误差；最好2个新闭环候选 | 必须保留W4自有scale/clip、其余W2无clip；不是坐标错误替换。当前A代码为准备状态 |
| L1 | attention与MLP、qkv/o/gate-up/down的恢复价值 | 选定阶段内BF16或W4 rescue，完整W2基线 | 模块组排序、Δteacher error/Δsuccess/理论额外bytes | 完整路径上复核；3选帧局部结果不能直接定全模型排名 |
| L2 | 量化前缀分布与校准覆盖是否解释clip冲突 | 教师输入与量化前缀输入；同样本/搜索预算；原与任务均衡多阶段校准 | 范围/clip饱和、逐层误差增长、action-token方向、闭环 | 改善仅搜索集不泛化则否证；探索32→128轨迹，不直接跳512大网格 |
| L3 | W3及敏感组G64/G32/zero-point的价值 | 独立W3profile；W2/W4匹配范围 | 成功—理论bits曲线、码利用率、metadata | W3未有位宽契约先补；每次只少量候选，保留原AWQ基线名 |
| V0 | 视觉G64组合是否可靠 | DINO G128/G64重新校准、SigLIP固定；语言固定可用配置 | 接口特征、动作、夹爪、未见闭环与branch图 | 语言恢复前只作次优先，勿继续盲目去视觉clip |
| P0 | 原OFT LoRA保留是否能抗W2 | Q2(Wbase+BA) vs Q2(Wbase)+BF16 BA | 精确base/adapter清单、merge parity、参数及闭环 | 原base来源未确认则停止该路线；从merged相减只能另名探索假设 |
| P1 | 固定码尺度能否恢复动作 | static no-clip、PTQ重校准、等预算scalePEFT、LoRA | q/z hash、梯度与保存parity、train/dev loss、各动作维度及新闭环 | frozen契约先过；开发误差改善且夹爪不坏才测试闭环 |
| P2 | 方向修复是否优于尺度修复 | rank4/8/16 Recovery LoRA、零/残差初始化、scale+LoRA、小affine | 等训练数据/步数/teacher/总param下Pareto，训练显存/时间 | 多轮离线收益不转闭环就研究margin/阶段，不无限加rank |
| M0 | 是否必须全W2，少量高精度的预算效率 | W2主体top1/5/10/20%模块W4或BF16、W4基线 | Δsuccess vs理论总bytes，后补packed实际bytes | 假量化显存增量不能作rescue score分母；准确命名mixed precision |
| C0 | 上下文专家是否互补 | 共享模型、固定专家、2–4专家、任务/离线最佳oracle | held-out context×expert热图，尺度谱、gripper与success | oracle gap小或最优尺度近乎相同就停Router；oracle仅诊断上界 |
| C1 | 路由收益是否真实且可部署 | Top1、随机/打乱context、等预算共享、固定专家、chunk固定选择 | 混淆/负载/切换次数、延迟、未见闭环，选错后果 | 不用future/teacher/GT推理；收益不覆盖成本则退回staticPEFT |
| F0 | 论文最终性能与效率 | BF16/W4/PTQ/最佳staticPEFT/候选，多suite对应checkpoint | 预冻结final、充分样本、训练seed，packedparity、P50/P95、全字节 | 开发和最终分开；没有real kernel就明确只得算法精度结果 |

这不是下一次开机要一次跑完的清单。G0后先L0-A与Q0，按结果选择分支；只派发有正确对照、输出位置和预计可完成预算的测试。

## 4. PEFT 小规模预实验的具体设计

### 4.1 训练目标与数据

先缓存BF16教师的normalized action chunk、raw action和选定action-token hidden，检查GT/教师标签区别。若能读取真实训练动作，核对7D表示、旋转、clamp、normalization及夹爪规则，再加示范损失。不要默认现有NPZ里的teacher动作就是GT。

建议目标从简单到复杂：`L = λa·Huber(aq, at) + λh·(1−cos(hq, ht)) + λg·Lgrip + λr·||δ||²`。λ仅开发集选择，各项先归一化量级，报告单项/组合消融。夹爪连续输出先做MSE/Huber或经验证的margin surrogate；不能对不具有概率语义的L1 head直接套BCE。teacher错误动作可能被模仿，因此最终以闭环和真实示范/任务表现验证。

训练预算先做可行性梯度一步→约200步pilot→有收益才1000步；batch1–2、gradient accumulation、选定模块与checkpointing。步数是探索安排，不是保证一张4090可按某时长完成。记录实际peak训练显存、每步时间与有效样本数，再决定后续预算。

### 4.2 最小训练契约

| 必过检查 | 意义 |
|---|---|
| 初始参数zero/identity时复现原PTQ前向 | 学习接口没有先改变基线 |
| 固定W整数q/z及排除模块hash不变 | 确认真正冻结；AWQ坐标必须明确 |
| 允许参数梯度有限且非零，禁用参数无更新 | 避免现有detach/no_grad造成伪训练 |
| 保存/恢复后输出与整数码一致 | 支持会话中断、备份、部署 |
| 模型总存储、optimizer、activation/teacher开销记录 | PEFT参数少不意味着早层反传显存小 |

最末action-token affine/低秩可先作为低成本可行性基线；若只有末端修复无法恢复控制，优先在L0/L1确定的语言模块做Recovery LoRA/scale。projector新增模块的梯度需穿过冻结LLM，不能套no_grad截断；也可能比末端修复更费训练显存。

### 4.3 研究分支如何收敛

若W4正常、W2只有大范围rescue才恢复，说明损伤可能分散，优先低秩/混精而非只学一层scale。若少量阶段恢复，优先该阶段静态PEFT。若静态有明显收益但不同任务偏好不同专家，再做C0/C1。若SQ T1正确、W16A4损失集中某token类型，才适合TSQ式激活映射专家。

若两位模型多个静态修复都不能恢复闭环，不应继续“Router必能救”的假设；考虑明确混精预算、稍高位宽或改变动作相关校准目标，并保留该否证结论。

## 5. 启动失败、空跑和会话收尾

后续由用户开机并指定实例，Codex直接SSH。启动前尽可能在本地完成代码/配置/输出规范；连入后顺序核对目标实例、GPU、必要数据、运行环境和最小入口，然后派发测试并确认进程及有效输出。

发生问题先就地修复：网络/公钥/环境激活/依赖/导入/文件路径/执行权限。Python单元测试失败与GPU入口失败分别定位；本地没有torch或WSL权限失败不证明服务器不能运行。不得以SSH失败推断无闲置GPU并盲克隆。

无法解决时保存错误、代码、已有结果及恢复说明，立即停止实验并关闭**本次确认的实例**；不能关闭旧014代替新107，不能销毁容器。优先SSH执行既有平台语义的关机工具；若SSH不可达，尝试只进行必要的控制台关机，不进行登录开机流程。若受平台认证/审批阻碍，立即明确告知用户未确认关机，不能声称成功。

不设置定时关机、不创建固定时间额度任务。每两短/一长检查五小时和周额度；任一<10%或下次预计无法完成即停止派发，预留至少3%完成备份/关机。读数不可用时承认未知，按实际会话消耗谨慎收尾，不能一直等未知读数恢复而计费空转。具体会话用户明确自己关机时遵从该覆盖。

每个测试均写持久盘独立目录、命令、stderr/stdout、状态/exit、配置与hash。程序等待/进程失败时要诊断或收尾，不能发送一句“继续”后结束会话而没有进程在运行。额度耗尽/客户端离线后无法保证还能操作，所有收尾必须提前。

## 6. 本轮待执行代码及时间依据

`scripts/run_awq_language_stage_rescue_a.sh`为L0-A：language-only W2无clip，分别回退blocks0–7和8–15到**独立W4profile**，视觉BF16，用32既有验证帧。`diagnostics/awq_interventions.py`修复了“W4 rescue不能同时W2去clip”的限制，保留W4自有scale/clip，只移除仍为W2目标的clip。其单元测试已扩展；本地通过不代表GPU已验证。部署使用独立overlay，不覆盖历史运行代码。

已有当前50状态BF16约9分钟、W4约9分钟，10状态W2约数分钟，这是特定运行的观察，不包含新profile搜索/训练，不能线性保证新测试时长。新107先G0，L0-A完成两短测试后核查额度；额度足够才做L0-B/Q0。长测试须预先定义有限输出，结束后立即核查。

本周末的报告应将新107结果追加为独立session并更新主报告“新证据”附录，保留既有结果、失败候选和运行版本，避免把计划表改写成已完成列表。
