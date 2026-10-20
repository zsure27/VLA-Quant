# OpenVLA-OFT 量化与微调实验框架

围绕多suite训练的OpenVLA-OFT，将 **suite选择 × 量化方法 × 微调策略** 组合成可追溯的小规模研究任务。默认单变量、多类探索；已有现象足够支持决策时收束，不要求每个候选跑满450。

## 主要入口

| 入口 | 用途 |
|---|---|
| [框架指南](docs/INFRA_GUIDE_CN.md) | 下次服务器开机后，从Git同步、配置、执行到统计的完整步骤 |
| [Ex_Reports](Ex_Reports/README.md) | 所有新旧实验：按任务归档，先整合结果分析，再列多轮日志 |
| [运行与备份约束](docs/EXPERIMENT_OPERATIONS_CN.md) | 服务器/SSH表、15分钟心跳、10%收尾、备份及必要关机 |
| [配置模板](configs/infra) | 多suite AWQ/SQ/BF16、LoRA/SVD/Routing绑定、额度和授权模板 |
| [统一代码入口](qvla/infra) | 组件注册、计划、运行控制、统计与任务报告 |
| [云端恢复入口](backups/github/README_CN.md) | 私有原始数据归档及恢复SHA；普通Git不下载大模型/视频 |

```bash
python -m qvla.infra list
python -m qvla.infra plan configs/infra/awq_multisuite.example.json --run-dir .local/preview-round --output .local/preview-plan.json
python -m unittest discover -s tests -p 'test_infra_*.py'
```

模板的全零SHA、占位路径与过期授权必须换成当前实际值；上述命令只编译计划，不加载模型或启动GPU。用户告知新任务和已开启的服务器后，才进入执行阶段。

## 能力范围

- Suites：LIBERO-Spatial、Object、Goal、10，各suite独立选择评测初态和数据清单。
- 量化：复用AWQ、SmoothQuant和BF16参考路径，允许注册其他方法。
- 微调：none、LoRA、Response-SVD LoRA、Routing统一策略接口。旧训练器仍有模型/数据/suite限制；新组合通过版本化命令绑定与必要数值门禁接入，不能把注册项当作已实测兼容。
- 统计：按cohort/suite/model输出JSON/CSV，严格配对比较，拒绝重复计数，保存全部失败、中断和成本。
- 运行：不可覆盖计划、当前轮授权、15分钟状态心跳、15%停止扩展/10%收尾、持久盘保护、GitHub同步与当前实例原生关机。

本次仅CPU验证框架和整理记录，没有开展跨suite模型实测。fake quant准确率不等于真实INT2/INT4部署加速。

## 已有研究结论

| 研究任务 | 结果 | 认识 |
|---|---|---|
| [普通与SVD初始化](Ex_Reports/20261015_普通与SVD初始化微调对照.md) | Z 2/450；S 426/450 | 当前配方对初始化敏感；SVD额外校准成本单列 |
| [零步补偿M0](Ex_Reports/20261016_零步SVD补偿与直接量化对照.md) | PTQ 77/450；S0 184/450 | 初始化本身有效，但恢复仍不充分 |
| [取帧时间B1](Ex_Reports/20261017_训练取帧时间覆盖对照.md) | Early 241/250；Phase 110/250 | 当前Phase取帧明显退化；已收束，不补450 |

三组不同运行不合并分母、不开历史旧计划，不跨轮相减当作因果训练收益；同十任务开发不称未见任务泛化，不读取offline_final_holdout标签。

## 目录与备份

`qvla/`保留数值后端，`qvla/infra/`组织实验；`configs/infra/`保存可移植模板；`Ex_Reports/`统一任务记录。原reports/results已迁入`Ex_Reports/_legacy/`并有逐项SHA映射，冻结证据不改。旧路径入口只留导航；历史脚本确需原路径时在服务器创建兼容目录链接，见框架指南。

公开 [VLA-Quant](https://github.com/zsure27/VLA-Quant)保存代码、分析、小结果和恢复索引；原始观测/动作、视频、参数和大日志保存在私有 [VLA-Quant-Archives](https://github.com/zsure27/VLA-Quant-Archives) Release。已授权备份迁移/本地副本清理完成；B5仍只有部分区间，不能称完整备份。

091历史运行已取得原生关机请求回执、心跳暂停；平台OFF/计费未独立核验。新框架不启动旧实验。第三方来源见[许可证与来源](第三方许可证与来源_CN.md)，整理前说明保存在[历史快照](docs/archive/README_BEFORE_CLEANUP_20261017.md)。
