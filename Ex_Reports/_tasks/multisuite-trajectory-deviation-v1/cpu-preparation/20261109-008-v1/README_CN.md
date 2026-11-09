# 008 无卡准备：结果与剩余门禁

用户于2026-11-09开启008无卡实例并授权准备；本轮没有启动模型推理、仿真回合或GPU任务，没有新增成功率或漂移结果。服务器保持无卡状态，准备完成后交由用户操作。

## 已通过

- 核验hostname为`autodl-container-1421458302-7454c1c8`、无GPU、cgroup内存上限2GiB；代码同步至起始提交`8c226f07f6321e7bac4d96f5e836e3c56aeefc04`。
- 四套GR00T检查点共29个文件、30,346,791,441字节逐文件大小/SHA校验通过；未重新下载模型。仅读取safetensors头部，未加载大模型。
- 固定GR00T、LIBERO、QuantVLA源码；另外预置GPTQ、OmniQuant、原DuQuant及已固定AWQ/SmoothQuant稀疏源码。精确提交与Python源码SHA见环境快照。
- QuantVLA实际运行脚本的include/exclude规则对应每suite **116个目标层**；不用注释中的估计数。四suite × DuQuant/QuantVLA × W4A8/W4A4共16份**不可执行**模板，绑定各自检查点收据、目标层与独立缓存身份。
- 上游DuQuantLinear小模块CPU合同通过：未量化变换最大绝对误差`7.152557e-7`；W4A8/W4A4输出形状和finite通过；三项保存重载输出逐位一致。这些小张量误差不代表全模型动作误差或控制成绩。
- 修复依赖后`pip check`通过，CPU旋转转换误差0，历史恢复视频102帧可实际解码。未复跑旧轨迹。

## 方法准备情况

| 方法／精度 | 本轮CPU准备范围 | 开卡前后仍需完成 |
|---|---|---|
| BF16 | 四套模型和原生环境核验 | 多suite实际输入／动作桥接及原生轨迹采集 |
| GPTQ W4A16、AWQ W4A16、OmniQuant W4A16、SmoothQuant W4A8/A4 | 固定上游源码；共用资产已就绪 | GR00T专用接入、目标层与完整算法合同；实际校准及全模型数值放行 |
| DuQuant W4A8/A4 | 原DuQuant与QuantVLA中的GR00T适配源码均固定；小模块和16模板中的对应8份就绪 | 不把GR00T适配自动等同原论文；全模型绑定、等价变换、校准与动作检查 |
| QuantVLA W4A8/A4 | 公开GR00T源码、ATM/OHB消费者位置、对应8份模板就绪 | ATM/OHB实际启用证据；按suite/位宽独立校准；完整模型与产物重载 |
| ActQuant W4A16 | 保留候选及既有来源审计 | GR00T公开实现适用性、导出／运行时与构建空间不足，未放行 |
| QVLA W4A16 | 保留用户要求的候选 | 现有公开实现偏OFT；忠实GR00T实现未证实，未放行 |
| HoloQ-VLA W4A4 | 保留候选 | 可复现实现及GR00T适配未确定，未放行 |

**不能据此称十二种量化已全部接入，或开卡即可直接跑完整队列。** CPU还能继续进行适配编码和小模块合同；真实激活/Hessian、搜索优化、全模型动作数值及显存测试需要GPU。

## 必须保留的限制

环境快照可用盘`11,049,193,472`字节（约10.29GiB），低于新方法12GiB门槛，差约1.71GiB；开卡前须解决空间并重新测量。没有删除现有模型、唯一原始数据或已保存结果，也没有购买空间。2GiB内存下不尝试加载全模型或重建大激活。

原QuantVLA脚本包含作者绝对路径与Long特定缓存，不能直接执行或跨suite借用。上游ATM数值不能未经验证挪作其他suite/位宽的校准结果。当前实现涉及fake quant，不能由它的耗时推断真实INT4内核加速。原生执行节奏保持GR00T每查询执行1步。

## 依赖修复与恢复

原decord轮子内部错误标为cp36，原pipablepytorch3d轮子内部标为cp311，而本环境Python3.10。第一次原轮子重装没有解决问题，失败日志仍在服务器。之后：

1. Decord仅重写WHEEL标签与RECORD；Python模块和共享库字节不变。可用仓库`scripts/repair_decord_wheel_metadata_v1.py`重建，输入固定SHA见收据；实际视频CPU解码另行通过。
2. 从PyPI官方`pipablepytorch3d-0.7.6.tar.gz`构建纯Python包，源SHA `1c9596c29f31ffa1acb867dd0c480eda77946958d9f5101c1dc96ad0db8e1cf5`：`PYTORCH3D_NO_EXTENSION=1 MAX_JOBS=1 CUDA_VISIBLE_DEVICES='' python -m pip wheel --no-deps --no-build-isolation --wheel-dir <new-dir> <verified-sdist>`，再无依赖安装。GR00T当前调用仅用`pytorch3d.transforms`；原生`_C`算子没有安装或验证，全模型桥接仍待GPU，不宣称整个PyTorch3D兼容。
3. 首次解码错误选中了历史零字节失败视频，实际拒绝；改用此前已审计的独立恢复视频通过。原失败和恢复身份均保留，没有改写旧成绩。

## 轮次记录与传输

`prepare_groot_cpu_v1.py`以独立screen执行，exit0，门禁为`PASS_GR00T_CPU_SOURCE_ASSET_AND_TINY_MODULE_CHECKS_ONLY`，始终`gpu_run_admitted=false`。完整运行日志与合成小模块产物留在008，公开只保存代码、配置、源码/资产身份和小收据。新下载主要是约86.72MB的官方依赖轮子/源码及轻量Git源码；未搬运30GB模型或历史轨迹。

取回公开证据压缩包仅20,611字节，SHA `45327e0d0f80c13b8a7aa49673b72b137a3c9298bd8e3da034818ee05d52ad2e`；21个文件逐项验证。清单在本目录`20261109-008-cpu-public-manifest.json`；它不等于完整模型或原始轨迹的备份，原始材料沿用已有私有归档索引。
