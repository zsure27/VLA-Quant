# B1 reset10–19启动

首50 Early48/Phase19、严格配对gate/实际三阶段exit0、完整分析及100视频双份全SHA通过。独立后片审计以真实首50复核全部NPZ/argv/seed/锁/视频，错误offset/count拒绝；CPU gate PASS_B1_TEMPORAL_SHARD_AUDIT_CPU。新候选仍同1000步，唯一GPU字段变化为offset/count/output/lock与对应原M0 PTQ task0/reset10首query参考路径SHA，其余argv/env/cwd逐项一致。不是按成绩选择续片。

预注册Git `562cb1d80c91b815f58c978151f86788638df10e` 已固定远端SHA，注册时primary剩57%/reset1792201971（22:12:54UTC当次API历史），GPUidle/8GiB盘/无STOP/前gate与分析和双份回执通过。

唯一新计划 `/root/autodl-tmp/qvla-repro/control/20261016-091-residual-resume-v2/B1-continuation-v1/reset-10-19/plan.json`，文件SHA `088af311b871c0e09d0667e586656ee71cf80c999c81488b00b6f845529ee0e5`，canonical `edb59271bf54ef8b02ca53332e62448752cf74d0f0c25baebbb2591b237fbdfa`。2026-10-16T22:13:33.446264+00:00 启动supervisor61487。顺序Early100→Phase100→strict audit，gate PASS_B1_TEMPORAL_FULL_SHARD_PAIRED_PROTOCOL。旧首50/微测/训练/M0/失败v1均不重跑。

输出R/B1-temporal-v1/reset-10-19/{early80,phase80}，gate R/B1-temporal-v1/reset-10-19-gate.json。下片offset20之前必须完整本片/累计150分析、失败成本、双份回执及索引，再fresh quota/disk注册。EV2 loader、独立审计注册器/本片lock和C/B1_TEMPORAL_FIRST50_ANALYSIS_CN.md已锁，不能活动覆盖；补充报告独立命名。
