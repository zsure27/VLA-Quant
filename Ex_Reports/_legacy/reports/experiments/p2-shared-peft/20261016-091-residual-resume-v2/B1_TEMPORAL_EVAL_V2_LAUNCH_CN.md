# B1评测v2独立恢复启动

v1真实首query协议失败，没有admitted query/正式回合，原0/10异常标签不作为成绩。完整双份失败归档89,285字节，SHA `fc946baa8bc9a9732dc7f00722f7a60743657bb3bb32b000b48ae190c995fa43`，35文件/36成员/10异常视频全SHA过。路径服务器/backups/091-residual-b1-temporal-eval-v1-failed-20261017-v1，本机D:/VLA-Backups/20261016-091-residual-resume-v2/同名。完整训练依赖另有976484948字节双份，未重跑。

CPU原PTQ trace证明float64归一化与原float32 NPZ存储表示之差；v2沿原capture转float32后严格array_equal，无容差。五种篡改字段全部拒绝；注入准入异常绕过官方逐episode Exception捕获，立即停止。CPU契约与负例PASS，预注册Git `479ade98e90a24ced86ee675651c506f00e0cd3e` 固定远端核验。实际相机与零动作匹配仍待GPU。

唯一新计划：`/root/autodl-tmp/qvla-repro/control/20261016-091-residual-resume-v2/B1-temporal-eval-v2/first50-plan-v1/plan.json`；文件SHA `65beb1b7a6bc8852345bab57998b5f41b65a6cb5b250dfa5fc5ba7a1c01a3a2c`，canonical `0f626a48df0c5a38db2d09e08fadc9a20a05c6fbf74fc2a07d1506c9b81f0310`。

2026-10-16T21:27:31.012403+00:00 启动supervisor52734。输出R/B1-temporal-v1/eval-v2-{micro,first50}/{early80,phase80}；门禁R/B1-temporal-v1/eval-v2-{micro,first50}-gate.json。按两侧micro10→strict audit→两侧50→strict audit，任何错误即停；原版本与训练/M0完成计划均不重跑，不覆盖锁。常规15/10/3，无新豁免。
