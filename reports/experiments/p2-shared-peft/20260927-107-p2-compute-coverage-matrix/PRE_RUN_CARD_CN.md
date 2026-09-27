# A/B/C/D 覆盖与计算量因子控制：预运行卡

时间：2026-09-27；分类：DEVELOPMENT/causal confound control；主阶段 P2.5，P4/P5 锁定。

HYPOTHESIS：data80 相对 data16 的固定观测改善同时改变了轨迹数和训练步数。B=16轨迹/1000步、C=80轨迹/200步与既有 A=16/200、D=80/1000 构成最小 2×2 因子控制，可区分覆盖、计算和交互；结果仅用于判断下一步是否值得闭环验证。

VARIABLE：两因子预注册矩阵仅补 B、C 两格。B 相对 A 只增步数，C 相对 A 只增轨迹覆盖。两阶段为同一受限因子实验，MAX_VARIANTS=2。

CONTROL：同 exact-12L AWQ-W2A16、语言 blocks18–19、rank8、Response-SVD、Smooth-L1 beta0.1、LR1e-4、优化器、seed7、同 batch 语义和现有 peft_train 轨迹切分。已核查 A/D manifest：seed、rank、初始化、loss、beta、LR 相同，差异为16/80轨迹及200/1000步。服务器沿用 A/D 的 probe.py 实现，不覆盖它。

PRIMARY_METRIC：冻结 trajectory-level router_dev 上每轨迹的动作误差中位数、改善轨迹比例与最差十分位。当前训练程序仍生成历史32帧回归诊断值，该值不用于门禁或最终结论。offline_final_holdout 封存。

DECISION：若 B 主要解释 D 的改善，优先审查优化目标/步数；若 C 主要解释，优先审查覆盖与访问状态分布；若两者交互明显，在 router_dev 上报告交互并只选一个预注册闭环候选。任何离线改善都不能直接通过 shared PEFT/P4 门禁。

STOP_DECISION：若 B/C 不改善 router_dev 或出现不稳定夹爪/最差轨迹退化，冻结该训练搜索，回到 P2.5 同观测误差机制。两个单元结束即停，不扩 rank、SVD 或新 loss；MAX_GPU_TIME=2小时，超过上限保留阶段产物并分析。

BACKBONE HASH：base W2 4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc；W4 ea3faca6130c88bd38fbe20be28eafc742d66605d339f8d9acfedb8da4703ea4；G64 947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8。checkpoint config bcb688a66f3e94a42311b77d37b7e1484886701f7b2aa8e36d165dfccac92ae6。具体校准轨迹哈希由运行 manifest 保存。

OUTPUT DIRECTORY：服务器持久盘 `/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20260927-107-p2-compute-coverage-matrix/`；本仓库同名 reports/results，本机完整归档同名 backups。预计 GPU 30–90 分钟；当前五小时剩余额度约63%，阶段边界重新读取。
