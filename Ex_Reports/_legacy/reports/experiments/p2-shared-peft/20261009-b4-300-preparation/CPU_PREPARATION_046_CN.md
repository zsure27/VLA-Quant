# 046 无GPU前置工作：B3/B4各300回合

日期：2026-10-09。当前046 hostname `autodl-container-90824aab47-6dc00ae3`，SSH端口38230；`nvidia-smi`确认为无设备。**本轮没有运行10步GPU烟雾、训练或闭环评测，也没有新增成功率。**

## 已通过的CPU门禁

| 门禁 | 实测结论 |
|---|---|
| 旧046服务器完整归档 | `/root/autodl-tmp/qvla-repro/backups/b2b3-046-final-20261009-0001`：13项主SHA及18,494项结果SHA通过，`PASS_PERSISTENT_ARCHIVE`。 |
| 本机完整副本 | 仓库忽略目录`.local/archives/20261008-046-b2-b3/b2b3-046-final-20261009-0001`：对应13＋18,494项逐文件SHA通过，`PASS_LOCAL_ARCHIVE_SHA256`。原`backups/experiments` junction指向不存在的D盘目录，因此本轮使用物理`.local/archives`，未覆盖或删除旧路径。 |
| 旧收尾与GitHub | 双份归档清单SHA、原`852d350` Git bundle、当前`zsure27/VLA-Quant`远端提交核验，`PASS_PRIOR_CLOSURE`；该回执不把上次用户手动关机记作平台OFF独立核验。 |
| 数据内容 | `PASS_DATA_SCOPE`：B3训练为十任务reset0–3的40学生回合/80观测；与已归档A4开发reset20–39的200回合在初态SHA与全部实际访问观测哈希上的交集均为0。十任务仍完全相同，分类仅为**同任务新初态开发恢复**。基础OFT/PTQ是否在任务层暴露的问题不由此门禁解决。 |
| 不可变计划 | 046独立checkout锁定`5cb45abef6a11ef7cf07bb8c2e94ff071da53b60`；四片`first50/next50/last100/final100`均通过结构校验，只有B3/B4两配置，实际offset/count依次为20/5、25/5、30/10、40/10。每配置正式50＋50＋100＋100＝300，微测不加总；277项运行输入/代码SHA再核为0不匹配。`PASS_B4_PAIR_V3_PLANS_REGISTERED`，`GPU_started=false`。 |

原始CPU回执、四片计划及锁定文件见同名[results目录](../../../../results/experiments/p2-shared-peft/20261009-b4-300-preparation/)。运行计划位于服务器`/root/autodl-tmp/qvla-repro/control/20261009-046-b4-pair300-v3r2/plans/`；对应会话输出根`/root/autodl-tmp/qvla-repro/backups/experiments/p2-shared-peft/20261009-046-b4-pair300-v3r2/`。计划快照的锁SHA为`5d5813ad04f8d1e2c805f9817da89bff5dc54ad2ef548db37c5ba01e8c085c77`。旧v2的200安排与一次no-GPU检测失败均保留历史，不混入v3的正式结果。

## 下一次有GPU后执行

先核对**当前实际hostname、GPU、持久盘空闲≥12GiB且保留6GiB归档余量、五小时实时额度与锁定277项SHA**。若实例身份或输入改变，应基于本次已推送版本重新注册，不复用旧hostname绑定计划。随后按计划顺序运行真实B4 10步烟雾、独立1000步训练、配对微测、首50硬审计，再按协议与资源依次完成50、100、100；每片只按协议门禁放行，不能按成功率方向择优续跑。任何数据交集/指令/初态/trace/锁失败都停止GPU。

主结论以本轮新鲜配对B4−B3的300回合rescue/break、任务分层95%区间及失败轨迹为准。A4/BF16/A0的历史匹配输入参照只有reset20–39的200回合，单列200子集；**不报告其300回合成功率，也不把开发初态称独立盲测。**完整结果后归档服务器持久盘及本机副本、推送并核验GitHub远端SHA，再关闭当轮实例。P4/P5/Router和`offline_final_holdout`继续锁定。

## 本次无GPU实例收尾

前置工作报告提交`f3d8e1c19bd1a4ee033f988e9bffb7bd2cce3b31`已由046上的独立Git网络路径核验为`zsure27/VLA-Quant`的`main`。046无GPU且没有待执行CPU工作，2026-10-09 13:43:20 UTC使用已同步SHA并通过完整归档/进程预检的专用helper发出原生supervisor停止请求，回执`execute=true`、pid751；SSH随后被远端关闭。详见同名[执行回执](../../../../results/experiments/p2-shared-peft/20261009-b4-300-preparation/shutdown-request-046.json)。**平台控制台OFF/停止计费未独立核验**。当前实例停止后，不自动开机或尝试运行计划；用户下次有GPU并告知实例后再继续。
