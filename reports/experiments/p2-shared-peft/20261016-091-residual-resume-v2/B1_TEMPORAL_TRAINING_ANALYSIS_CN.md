# B1时间覆盖训练完成分析

六阶段全部exit0/revision20，20:49:35UTC完成。PASS_B1_TEMPORAL_SMOKE_PAIR及PASS_B1_TEMPORAL_TRAIN_PAIR，两个版本均19988480参数、448初始因子逐值一致，底座/目标描述/优化抽样槽位顺序完全一致，底座冻结SHA保持36d41eb329862e8c8a6f302a88655b729b85ebe750aaf93dc32de92b2ba984b6。真实零分支、全部期望梯度finite、保存重载动作完全一致通过。

## 实际训练与成本

Early80与Phase80各从同一训练前SVD因子独立重训1000步，之前10步不累计。Early80全80教师/处理输入逐位桥接原缓存；Phase80使用自己的新鲜教师，首末重复query逐位一致，两侧各82教师前向。新数据均来自同40完整A4训练轨迹，1020原查询逐SHA，数据角色不变，无成功过滤。

Early正式用时353.827s，Phase正式355.406s；两侧烟雾分别128.800s/128.420s。正式peak CUDA均19310084608字节。每侧适配器39976960 BF16 tensor字节；保存checkpoint/初始/教师/日志带来额外归档容量，不把训练文件总量冒充部署模型大小。墙钟含标签/加载/校验/冻结hash/保存，非纯前向速度，不称INT2真实内核加速。

Early最后一个训练样本的SmoothL1=0.00637100，Phase=0.03112607。它们是各自不同时间观测上的单步训练损失，不能直接比较数据集难度、均值泛化误差或闭环成功，不能据此挑checkpoint。两侧统一使用预注册1000步。

Early adapter SHA 459174ac508397133f3a323e85189b84c065d53637535af0985efe0574bae2ac；Phase bc7ed6751e13f0e8c8c109ae1ed7efa55b4f011fdaa9336a721b50f0f3d565f1。完整artifact/逐步trace/来源在同名results/B1-training-complete-records.json，简要成本B1-training-summary.json。新版本b1-temporal80-v1-20261017不与历史B1/B3身份混同；旧046 S426/450仅历史参照，不合并作因果训练增益。

## 后续硬门禁

本阶段只有训练/结构证据，未得新闭环结果。完整双份训练归档须包含全部新训练参数/教师/160原始输入/1020原始来源查询/代码/命令/退出及门禁；未来评测排除。随后用独立artifact验证器核验新version/variant/1000步/源SHA/初始/训练角色/参数/冻结/配对gate，再接真实首query一次本体归一化、保存重载、零分支与匹配PTQ首动作的数值桥接、两侧micro10及配对gate、两侧首50及硬gate。不改变损失或量化/初始化，不读holdout，不扩routing。

## 额度自然刷新已确认

2026-10-16 20:52:52UTC实时primary used2%/remaining98%，resetsAt从1792183963变为1792201971。自然刷新已经接口实际确认，首次窗口豁免到此终止、不续期。以后恢复剩15%停扩展并完整双份/Git核验，剩10%保存终止/确认GPU闲/原生关091，留3%；最终原生回执/PAUSED/完整总结后正常关Windows，不强制应用。