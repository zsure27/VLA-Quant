# v4：补充OpenVLA-OFT的同口径可视化

## 结果与分析

已在v3基础上增加第三模型OpenVLA-OFT的六类图，以及三模型并排总览。当前仍无服务器、无新模型运行。**全部新增图为固定随机种子构造的标量分布示意，不是OFT真实轨迹或论文实验结果；不能把示意差异解释为三个模型的实际优劣。** 不用历史Spatial的M0/B1分母补充此次四suite实测。

- [三模型总览](preview-v4/index.html)：π0.5、GR00T-N1.5、OpenVLA-OFT分别占一行/组。
- [OFT独立六组图](preview-v4/oft/index.html)：放大检查联合分布、中心、离散度、下一query响应、四suite共性矩阵和有效样本数。
- 原有[v2空间轨迹与同步播放器](preview-v2/index.html)继续作为OFT基础展示；其中合成轨迹不能当作新采集结果。

最新范围为三个模型 × LIBERO四suite × BF16/ActQuant-W4A16/SmoothQuant-W4A8/QuantVLA-W4A8。OFT新增首批32自然回合，三模型首批合计**96回合**（3×4×4×2任务×1初态），不是立即执行目标；可按前述覆盖、资源与停止规则分片。此前两模型64回合预算保留为v3历史。

所有方法名称均为研究比较目标。OFT已有BF16/AWQ记录器与SQ接口，但本轮四suite实际输入/调度等价、SQ数值合同仍待GPU验证；ActQuant和QuantVLA的OFT适配尚未实现，不把其论文在π/GR上的兼容性直接迁移到OFT。未通过的组合保持缺失并说明原因，不换成AWQ却沿用ActQuant标签。AWQ-W4/W2为可选、独立OFT压力对照，不悄悄加入主表或混入历史SVD/LoRA训练结果。

## 原生执行跨度与可比性

| 模型 | 本轮预登记每query执行步数 | 查询后5次对应控制步数 |
|---|---:|---:|
| π0.5 | 5 | 25 |
| GR00T-N1.5 | 1 | 5 |
| OpenVLA-OFT | 8 | 40 |

8步来自当前OFT评测配置约定，不是声称所有OFT部署都固定8步。真实起跑须核对checkpoint、action head、实际argv和调度身份；不匹配则拒绝当前协议，另版本登记。v3测量合同保持原字节，不修改为默许OFT。

v4图保留各模型原生query轴，并标明乘以5/1/8的控制步数；联合分布标题标出对应控制步。三模型相同query序号并不等于相同物理时间或动作阶段，不能直接比较这些曲线的斜率来排名模型。每模型独立冻结尺度，主要比较本模型量化相对BF16的效应方向，再检查是否跨suite重复；不把跨模型归一化误差的绝对大小解释为相同毫米/度精度。

可选短扰动探针若扩展到OFT，新增48条，三模型共144条（3模型×4suite×3精度×4正负方向），无扰动参考36格。只有完整重放门禁通过才能复用自然回合，否则最多另加36条短参考；它们不是144个完整任务，前缀重放、加载和额外预测均计费。保持注入后40控制步上限：三模型公共query展示网格为0–5；真实采集另保留各自全长。此处缩短查询网格用于展示，不改变自然回合长度或重规划频率。没有授权服务器时不会派发这些预算。

## 代码与验证

- `qvla/runtime_error_protocol_v4.py`：独立增加OFT h=8身份，保留同精度、t→t+1、六维运动/夹爪分离、有限值及噪声身份检查。
- `diagnostics/runtime_commonality_v4.py`：支持三模型自动分面及`--model openvla_oft`独立视图；实测数据必须带v4内容绑定收据。不会读取旧论文表格或历史成功率生成假轨迹。
- `configs/infra/runtime_commonality_v4/study.json`：不可执行的研究规格，模型/方法适配与本轮GPU门禁缺失时阻断。
- 新增3项CPU合同测试：OFT实际计入第8动作位、排除未执行尾部与夹爪；拒绝跨模型/错调度；验证冻结v3仍拒绝OFT。模型真正兼容性与可恢复扰动采集尚未验证。

```bash
python -m unittest discover -s tests -p 'test_runtime_error_protocol_v4.py' -v
python -m diagnostics.runtime_commonality_v4 --demo --output .local/runtime-three-models-NEW
python -m diagnostics.runtime_commonality_v4 --demo --model openvla_oft --output .local/runtime-oft-NEW
```

仅代码、规格、分析、小收据和合成PNG/PDF增量进入公开Git；原始观测/动作/视频/权重仍按私有备份规范。此次不调用服务器，不修改历史计划，不启用心跳，不关闭Windows。

## 轮次日志

- prepare-cpu-v4（2026-11-06）：按用户要求增加OFT可视化和独立视图，保留v3源码与图原字节；新增CPU合同检查和图形生成。当前实测回合仍为0。
