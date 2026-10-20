# data80 Recovery-LoRA 稳定性试跑与 P2.5 重对齐

## 目的与证据等级

本轮完成正在运行的 exact-12L、blocks18–19、rank8 Response-SVD、Smooth-L1 data80
Recovery-LoRA 50 回合配对闭环。根据 2026-09-26 修正，该分片只作为 **closed-loop
stability pilot**：相同 slice 的 12L 与 14L 都是 43/50，没有观察到静态 W4 headroom，
不能用它估计 W4 恢复率，也不是最终盲测。

## 结果

| 配置 | 成功 | 相对12L |
|---|---:|---:|
| C0 exact-12L | 43/50 | — |
| C1 exact-12L + data80 Recovery-LoRA | 42/50 | -1/50（-2 pp） |
| C2 14L static | 43/50 | 0/50 |

C0/C1 配对四格为共同成功40、仅12L成功3、仅LoRA成功2、共同失败5；exact McNemar
`p=1.0`，配对 bootstrap 的 LoRA−12L 95% 区间为 `[-10 pp,+6 pp]`。结果没有灾难性
退化，也没有可信正收益。由于 `C2-C0=0`，`recovery_fraction=null` 是唯一有效写法。

data80 的固定观测结果仍为 normalized action MSE 0.02284、median paired delta -0.02198、
31/32 帧改善、gripper disagreement 0.05078。它与闭环 42/50 合在一起形成清晰的 Case B：
**很大的固定观测教师误差下降没有转化为闭环成功率改善。**

## 两个混杂因素的修正

1. data16 使用16条轨迹、约200步；data80 使用80条轨迹、1000步。每个选中帧的有效曝光
   都约12.5次，但总优化量增加约5倍，因此不能把提升单独归因于轨迹覆盖。下轮只补缺失的
   B=16/1000 与 C=80/200，再与 A=16/200、D=80/1000 在轨迹级 `router_dev` 比较。
2. 当前 slice 的 12L=14L=43/50，不适合回答 blocks18–19 升 W4 的恢复比例。历史上
   12L失败而14L成功的 episode 只能作为 post-hoc 机制子集，不报告为无偏成功率。

## P2.5 与动作语义

评测器源码已核实：模型一次预测8步，全部8步从队列执行完后才重新推理，属于 open-loop
chunk。新的 P2.5 实现将保存学生实际访问的主/腕相机、proprio、指令、学生8×7动作和
H17摘要；学生退出后单独加载 BF16，在完全相同的序列化观测上查询教师，避免两套7B模型
同时驻留。逐 query 报告位置、旋转、夹爪连续误差、夹爪阈值/margin和8个执行位置。
未来成功只作分析标签，不作模型输入。

历史32帧只保留为回归集。新的 `router_dev` 抽样对59条轨迹使用固定归一化位置
0.1/0.3/0.5/0.7/0.9，并按轨迹汇总 median、改善比例和最差十分位。`offline_final_holdout`
继续封存。

## 成本

- 当前 LoRA checkpoint：1,249,280 参数，实际序列化 **2,509,154 bytes**。
- Scale-PEFT：6,324,224 个 FP32 参数，原始张量至少25,296,896 bytes；实际 checkpoint
  bytes 仍需从原件审计，当前不把参数数等同于序列化大小。
- blocks18–19 的14个 Llama Linear 合计约404,750,336个权重；理想纯权重 W2→W4 增量约
  101,187,584 bytes，尚未计 scale/zero 和容器开销。LoRA 实际文件约为该理想增量的2.48%。

## Gate 结论

Recovery-LoRA 尚未通过 shared PEFT 闭环 gate。不得进入 P4 专家或 P5 Router，也不扩大
rank/SVD/目标层。下次开机的顺序固定为：

1. 审计12L的411/412与14L的430/431来源，确定唯一 canonical 口径；
2. 建立轨迹级 `router_dev`；
3. 运行小规模 P2.5 同观测诊断；
4. 补 B/C compute-vs-coverage；
5. 若学生访问分布误差显著增长，先做同rank8预算的 student-state teacher relabeling。

详细预注册与停止条件见
[`P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md`](../../../../docs/P2_5_ON_POLICY_ALIGNMENT_PLAN_20260926_CN.md)。

## 复现与备份

- 服务器结果：`/root/autodl-tmp/qvla-repro/eval/p2c-lora-closed-loop-pilot-0-4-20260926-120417-102671`
- 服务器持久盘归档：`/root/autodl-tmp/qvla-repro/backups/p2_data80_stability_20260926-123704-852d350`
- 本机完整归档：`backups/experiments/p2-shared-peft/20260926-107-p2-data80-stability-and-realignment/`
- Git 小结果：`results/experiments/p2-shared-peft/20260926-107-p2-data80-stability-and-realignment/`
- 归档包含150个 rollout 视频、三组日志与策略查询、代码 bundle、patch、恢复说明和大文件哈希清单；普通 Git 只保存配对小结果、命令、哈希和本分析。

