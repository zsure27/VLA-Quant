
**当前无服务器，无GPU或新实测轨迹。** 已参考用户35页PPT完善权重/激活对照与可视化；新增7组PNG/PDF与控制步同步播放器，加上v1共13组展示。全部预览为独立构造的合成示意，不是PPT或LIBERO实测结果。PPT原文件未改动，未上传公开Git。

v2先跑BF16、AWQ-W4A16/W2A16的W阶段120条；SQ等价及同W4门禁通过后，才允许SQ-W4A16/A8/A4的A阶段另120条。四suite各10任务全部覆盖，首期上限240条。纯激活与W/A四分支先做离线小样本；不默认全部方法全网格，不训练。此预算替代下方v1的原始扩展安排。

新增功能：同输入W/A/交互项分解、有符号动作变化方向投影、chunk误差遮罩、事件对齐缺失计数、激活统计，独立v2运行门禁与6份阻断模板。7项新CPU测试、原诊断7项和原框架7项均通过。实际SQ等价、全模型四分支重放、逐层捕获、视频接触同步仍需服务器阶段独立实现或验证。

- [完整v2设计与PPT参考页](./_tasks/multisuite-trajectory-deviation-v1/DESIGN_V2_CN.md)
- [v2同步轨迹与七组图](./_tasks/multisuite-trajectory-deviation-v1/preview-v2/index.html)
- [v1六组图](./_tasks/multisuite-trajectory-deviation-v1/preview/index.html)
- [v2 CPU收据](./_tasks/multisuite-trajectory-deviation-v1/runs/prepare-cpu-v2.json)
