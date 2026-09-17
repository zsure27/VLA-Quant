# 本轮AWQ基线原始小文件

只包含正式完成的配对片；EVAL文本为原始未改写日志，paired-results由CPU解析器生成。每片有命令、冻结source/profile hash、退出码、源码快照和动作trace。trace为各自闭环状态下的原始策略chunk，不能将分岔状态的动作差当同输入量化误差。500回合尚未补齐，旧开发成绩不混入。

报告和图表见[本轮报告](../../reports/sessions/20260917-107-baseline-validation/README_CN.md)。模型、profile二进制和原始视频不在普通Git中，完整视频双份备份另有清单。
