# 当前任务摘要（v3）

无服务器、无GPU、无新实测结论。主范围为π0.5/GR00T-N1.5 × LIBERO四suite × BF16/ActQuant-W4A16/SmoothQuant-W4A8/QuantVLA-W4A8。先64自然回合；可选短扰动探针独立注册，不跑默认450或方法全网格。π/GR原生采集与量化适配尚未实现，不能复用OFT绑定冒称兼容。

- [完整v3设计](DESIGN_V3_CN.md)：同精度扰动分支、误差t→动作t+1、原生5/1步、风险集、预算与限制。
- [v3六组图](preview-v3/index.html)：联合分布、中心、离散度、响应、共性矩阵和风险集。
- [v2七组图和空间轨迹播放器](preview-v2/index.html)：空间偏移、W/A分解、动作块及事件对齐。
- [v1六组图](preview/index.html)：历史基础视图。
- [v3 CPU收据](runs/prepare-cpu-v3.json)：新增7项CPU检查；所有图为合成示意。

完整同任务日志在[任务卡](../../20261106_1504_四套件多任务轨迹偏移诊断.md)。v1/v2历史方案不再作为主起跑指令，不覆盖冻结源码或GPU门禁。
