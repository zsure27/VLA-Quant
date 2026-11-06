# 权重与激活量化轨迹诊断 v2

2026-11-06，根据用户提供的35页PPT修订。同一研究任务沿用原task_id和任务起始时间；v1源码、配置、预览、CPU收据保持原字节。当前无服务器，GPU/闭环/微调次数均为0。下述图形预览全部为独立构造的数学合成数据，没有引用PPT中的数值作为本项目结果。

## PPT中采用的内容与迁移边界

| PPT页 | 采用的分析方式 | 本项目处理 |
|---|---|---|
| 2–7 | 方法×W/A精度×benchmark的分面矩阵 | 四LIBERO suite分列，未运行格留空；只把已接入方法生成可门禁配置，不填造成功率 |
| 9、29 | 校准量、absmax/p99.9、跨batch覆盖 | 独立校准角色上比较固定80/160观测等预登记预算；阈值/统计配方单变量，不照搬源PPT的饱和样本数 |
| 10–12 | 动作几何、逐维差异、CDF、最大差异局部 | 分开平移/旋转/夹爪，记录chunk位置，增加真实空间路径与控制步局部放大 |
| 15–18 | 激活离群、峰度/偏度、局部W/A耦合 | 同输入、同等价坐标做四分支分解；当前新增CPU统计和输入接口，未声称已捕获所有模型层 |
| 19–28 | 视觉/语言/动作模块误差与权重结构热图 | 首轮固定动作Head/proprio/embedding/norm高精度；模块恢复为后续单变量，OFT无π0.5式DiT，按实际named_modules建立清单 |
| 31–32 | 误差沿动作变化方向投影、同W4下A开关消融 | 保留有符号平行分量，排除夹爪，先固定6维尺度；把该方向称局部动作变化代理，不直接称真实模式边界或安全失效 |

PPT主表含π0.5及LIBERO-Safety/VLA-Arena，不是本项目OFT四suite实测。AWQ/GPTQ原生主线是权重量化，表中的AWQ-W4A8须查清额外激活QDQ定义、校准和保护范围，不能直接套上现有AWQ后端。QVLA、QuantVLA、Ω-QVLA名称分别核对。源PPT的“量化边界”“视觉不起作用”“模式跳变导致失败”等为需验证解释，不作为本项目已证实机制。校准只选成功轨迹会有选择偏差；本项目合法角色内保留成功与失败。OFT的视觉/语言层含大量token，不能从其他模型推断“一次前向只有10个token”；token数也不等于独立状态数。

源文件SHA256：c7a609f5d94275b99ba1ee5f772803ae078dbc208b147a461d99fe875ef449ef。PPT及其图片、文本提取只在本机私有工作区阅读，不上传公开Git；源文件保持不变。

## 方法清单与有界实验预算

当前后端可复用AWQ、SmoothQuant，但两者多suite真实GPU门禁均待实测，SQ还须先解决历史只平滑等价失败。GPTQ、OmniQuant无LET/有LET、DuQuant、QuantVLA列为候选；Ω-QVLA先确认确切来源。**清单不等于实现，更不等于通过兼容性验证。** W3仅作待适配方法点，不把既有W2/W4/W8实现改标签冒充W3。

| 阶段 | 条件与目的 | 预算与停止 |
|---|---|---|
| G0 | 新联合模型身份、输入/原始本体一次归一化、记录器开关等价；SQ W16A16只平滑控制 | 必要的小probe；C0不过不启动SQ长轨迹 |
| W | BF16、AWQ-W4A16、AWQ-W2A16 | 四suite各10任务×初态5=120条；保持实际保护范围和配对协议 |
| A-offline | SQ-W16A16、SQ-W16A8/A4与SQ-W4A16/A8/A4 | 合法开发轨迹中按预登记规则取每任务2个query，共80状态，每档前向另计；不是80条完整闭环；C0先通过 |
| A-closed | SQ-W4A16、SQ-W4A8、SQ-W4A4 | 相同W4权重与平滑坐标、同profile；四suite各10任务×初态5=120条，须离线/数值门禁通过 |
| Confirm | 最多两个按预定规则存活的候选与BF16 | 初态6/7最多240条；总初期闭环上限240，不默认扩成全方法×全位宽网格。确认规则在读6/7结果前冻结 |
| Optional | 单个GPTQ/OmniQuant/DuQuant对照，或局部模块恢复/校准策略变化 | 每项独立预算和版本；先离线，再决定是否40任务小片，不自动执行 |

一次W位宽对照可能需要各位宽独立搜索scale/clip，应报告它是完整量化配方差异；一次A消融必须共享同一个W4内容SHA、目标模块、平滑因子和保护范围。不能以不同方法各自搜索出来的W4当成同W4激活开关实验。核心先不训练，防止LoRA掩盖量化/重参数化问题。

## W/A误差分解及轨迹口径

对同一观测、相同动作表示、同一等价坐标，记f00为W16A16（SQ只平滑已证明等价），f10为W4A16，f01为W16Ab，f11为W4Ab：

```
delta_W = f10 - f00
delta_A = f01 - f00
interaction = f11 - f10 - f01 + f00
total = delta_W + delta_A + interaction = f11 - f00
```

该式是输出向量的精确对照分解，不是误差范数相加，也不是训练/闭环失效的唯一因果归因。四分支可以有非线性传播；interaction只描述所选对照的非加性。原始AWQ与SQ坐标不同不能混进一个四分支。代码拒绝输入SHA和coordinate_id不一致。

轨迹同时保留：

- 同输入的逐维命令差，教师重标学生真实访问的状态；query与控制步、chunk位置明确映射，尾部未执行slot留空。
- 配对初态的独立闭环末端路径与姿态差，只称执行后果的描述性偏离。空间路径用m/mm，姿态用世界坐标SO(3) log和测地角，动作命令单位另列。
- 夹爪关闭“指令”事件与实际抓取/接触事件分开。v2实现关闭指令检测和不完整窗口计数；接触、物体抬升、物体/末端同步的视频真值采集仍需环境绑定。
- 局部动作变化方向u取同query教师动作块相邻6维命令的变化，经冻结尺度标准化；error沿u投影保留正负，角度0–180°，零变化/零误差不伪造0°。该指标不是接触边界、神经模式分类或安全证明。末端反向运动、方向对齐与失效之间要做窗口干预验证。

“大动作差异步”阈值来自独立重复性/控制容差的冻结校准，而非照搬PPT的mm阈值。所有图都给数据类型、参考、坐标、单位、缺失与分母；不挑成功回合，不用相关控制步扩充样本量。CDF只描述相应的观测/slot群体，不冒充独立episode置信度。

## 新增展示与使用方法

新增7组PNG/PDF（加v1六组，共13组），以及一个可拖动控制步的本地浏览页：

| 图号 | 展示 | 防止误读 |
|---|---|---|
| 07 | 三维末端路径、XY/XZ投影、指定窗口放大 | 同一单位，焦点来自审计窗口；不是只看误差差值的抖动线 |
| 08 | XYZ有符号位置偏移与有向旋转分量 | 保留方向，不把独立闭环差异全归因于量化 |
| 09 | W-only、A-only、interaction、total | 四分支同输入同坐标；阴影是slot范围，不是置信区间 |
| 10 | query×chunk位置误差热图，共享色标 | 区分后半chunk累积与query间突变，未执行slot遮罩 |
| 11 | 有符号平行/垂直散点与方向角CDF | 称动作变化代理，不贴“灾难/安全”标签 |
| 12 | 夹爪关闭指令对齐的轨迹差，逐偏移步分母 | 无事件回合单列，不把关闭指令称成功抓取 |
| 13 | 激活分布/冻结范围及A8/A4重构误差CDF | 单元素统计不冒充独立样本；实测须提供真实QDQ重构 |

同步页上选择量化条件、拖动控制步，可同时看BF16与量化末端位置和误差游标。当前页只放合成示意，尚未接相机视频。真正图像/视频/接触同步将以实际step/query索引接入，不能用示意机器人图代替实测视频。

```bash
python -m unittest discover -s tests -p 'test_trajectory_diagnostics_v*.py' -v
python -m diagnostics.trajectory_study_v2 --demo --output .local/NEW-v2-preview
# 实测JSON bundle的结构见 synthetic_bundle()，但须填真实采集/教师结果、执行slot mask和独立分析门禁。
python -m diagnostics.trajectory_study_v2 --bundle PRIVATE/audited_bundle.json --output PRIVATE/NEW-analysis
# 配置中SQ使用显式诊断alias，plan/run都要加载同一个插件。
python -m qvla.infra --plugin qvla.trajectory_infra_v2 plan PRIVATE/filled.json --run-dir PRIVATE/ROUND --output PRIVATE/ROUND.plan.json
```

`configs/infra/trajectory_study_v2/`有6份闭环阻断模板、6份身份模板、方法状态和分阶段预算。离线C0/激活-only不是可直接启动40回合的模板。`qvla/run_trajectory_diagnostic_v2.py`复用现有官方源码适配，逐一核profile内容SHA，并要求SQ等价门禁；不是新实现8种量化后端。新版本运行门禁与v1分开，不继承旧GPU通过事实。

当前准备的边界：4分支全模型重放/逐层重放的完整GPU调度器、所有层的自动feature捕获、跨benchmark数据适配及视频接触同步仍未实现或验证；只提供CPU数学/分析接口与已存在AWQ/SQ的阻断执行绑定。后续服务器先实现/验证必要的同输入重放和小范围层探针，不为了展示自动缓存整层激活。可先8个代表模块×4行/调用、总行缓存64MiB上限；记录token取样规则及实际调用次数，不套用PPT的420层编号。

## 研究约束与备份

四suite均为LIBERO开发任务，联合训练覆盖不能称未见泛化。动作敏感初始化、OPD、在线路由仍是后续独立方法，首轮不同时改动。默认15分钟只读心跳、剩15%停扩展、10%收尾、留3%；新方法>=12GiB、续片>=8GiB、运行留6GiB，单模型顺序。暂无服务器，不创建运行心跳、不启GPU、不关Windows。

公开Git只上传新代码/配置/聚合分析/明确标识的合成预览，不上传源PPT或真实原始数据；真实观测、动作、视频、checkpoint依旧进入私有归档，传输前统计新增字节，默认每批2GiB，分块去重验证。新增预览约1.5MB以内，禁止为此搬运历史实验全量包。

方法原始来源：[AWQ](https://github.com/mit-han-lab/llm-awq)、[GPTQ](https://github.com/IST-DASLab/gptq)、[SmoothQuant](https://github.com/mit-han-lab/smoothquant)、[OmniQuant](https://github.com/OpenGVLab/OmniQuant)、[DuQuant](https://github.com/Hsu1023/DuQuant)。注册状态与门禁以本项目代码/收据为准，不以论文有实现推断OFT兼容。
