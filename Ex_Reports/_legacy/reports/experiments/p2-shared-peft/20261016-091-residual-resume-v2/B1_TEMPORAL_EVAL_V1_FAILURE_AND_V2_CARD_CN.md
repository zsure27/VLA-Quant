# B1评测v1协议失败及独立v2卡

v1 micro-early80在真实首query桥接失败，revision4/exit1，后续未启动，实际GPU已空闲。全部10个episode尝试无通过query，不能将日志0/10作为模型成绩，正式分母0。旧plan/code/output保留，训练与M0不重跑。

CPU用原PTQ实际raw trace重算：归一化float64直接与记录float32不等，最大差2.398526843805371e-08；按原OnPolicyObservationCapture的float32存储后逐位相同。v1桥接遗漏此存储转换。新v2比较严格沿原capture表示，仍array_equal无容差，双相机/任务/动作同样逐位比较并保存每字段检查和实际NPZ。实际新相机/动作匹配尚待GPU验证，不能由CPU推断通过。

v1官方run_episode捕获Exception后继续了其余task，最后外层因无admitted query exit1。v2仅在首query admission异常时抛独立BaseException，绕过该逐episode捕获并由外层finally清理；首错即整计划停止，CPU注入测试验证。其他评测异常仍由严格audit拒绝。

v2独立C/B1-temporal-eval-v2，唯一新输出R/B1-temporal-v1/eval-v2-{micro,first50}，新版本b1-temporal80-eval-v2-20261017，两侧同1000训练产物/数据/算法/seed与预算不变，重复仅为无有效query的协议修复。Early微测10→Phase微测10→strict audit→各50及audit；运行12GiB/留6GiB/前gate/预注册Git/新鲜quota必须通过。不扩routing、不读holdout、不按成绩方向选择。
