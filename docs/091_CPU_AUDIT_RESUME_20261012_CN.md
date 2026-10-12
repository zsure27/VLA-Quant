# 091 无卡审计续跑记录

## 当前目标

先审阅并解决BF16/AWQ-W4与QVLA表2的差异，再完成B5前置。无卡阶段不启动GPU烟雾、训练或真实策略评测；完成后通知用户开卡。

## 已核验连接与资源

- SSH：root@connect.nmb1.seetacloud.com:18575。
- hostname：autodl-container-dfbc4bb0ff-1e85dd94。
- 用户经平台终端提供并核对的ED25519：SHA256:y2DhrdHAWfTCoY9uLz0ve3EzTKYtWNBBWm80QUTOkXg。
- 专用密钥：C:/Users/zsure/.ssh/id_ed25519_vla_014；隔离known-hosts：C:/Users/zsure/AppData/Local/Temp/vla-known-hosts-091-audit-20261012。
- Python：/root/miniconda3/envs/qvla-oft/bin/python，torch2.2.0+cu121 / transformers4.40.1；CUDA不可用。
- cgroup内存2GiB、CPU0.5核；不得完整加载7B CPU模型，使用流式权重/小模型契约。
- 当前持久盘约11GiB空闲；GPU进程未运行。
- closure-tools双helper SHA与编译通过；尚未执行关机。

## 保存时的完成项

1. 历史A0来源、实际422目标与500/450配对指标重算通过。
2. 本机9项CPU契约通过；原临时目录权限问题已用工作区目录解决。
3. 表2的92.5%为四suite平均，Spatial应比较93.0%；BF16仅差0.2pp，A0差4.2pp。
4. QVLA未公开表2AWQ全部配方；当前差异包括视觉自写适配、32观测校准和评测随机流。尚未证明哪项导致成功率差。
5. B5 PRE-RUN CARD已写；bank/state/features及真实协议探针由子任务新增代码，保存状态需看Git文件与测试回执。

## 下一步

- 读取真实profile二进制并流式核验checkpoint SHA、422名称/形状与整数码/重建公式；不能用metadata代替。
- 用真实run_task/run_episode处理链比较pristine upstream、历史paired、修正paired的连续reset观测/输入哈希；协议分别重复，失败关闭。
- B5完成registry/因果输入/训练数据角色合同、训练和运行入口、不可变计划以及CPU数值/状态契约。既有哈希绑定B3/B4代码保持字节不变。
- reset5–49为同任务开发450，训练0–3、router dev4；PTQ暴露另查。不能宣称独立盲测。
- 真实GPU前向/烟雾及成功率因果归因待用户开卡，不能用CPU PASS替代。

## 本次额度覆盖

用户指定北京时间2026-10-12 15:16刷新：首次约15%/10%阈值只保存并保持无卡091，等待实时五小时窗口实际刷新。记录旧窗口 resetsAt=1791789403 和 usedPercent=86 仅为保存时历史，后续不能冒充实时。确认刷新后继续，下一窗口约15%完成归档/本机/Git，约10%停止新工作并原生关当前091，留至少3%；任务完成亦可收尾。本轮不关闭Windows。

低频心跳每15分钟；未刷新时只读实时额度，状态不变静默，不反复长读取或频繁唤醒。无卡等待刷新是用户明确授权；不可修复的问题仍须清楚报告。
