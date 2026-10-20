# P2.5：学生状态重标候选的配对闭环（预注册）

状态：PRE_REGISTERED，开始GPU闭环前冻结。候选为exact12L mixed backbone、仅语言blocks18–19、rank8 Recovery-LoRA，原始peft_train80的Response-SVD初始化不变；唯一训练变量是80个BF16重标的学生访问状态，来自每任务4个初态（reset0–3）的query0–1，1000步和uniform Smooth-L1不变。该候选的adapter SHA与评测器/profile哈希必须保存。

仅评价从未用于该候选训练的Spatial初态5–49：10任务×45=450回合。之前冻结的相同评价器、同初态seed、同checkpoint/profile的12L(C0)、旧demo-only LoRA(C1)、静态14L(C2)原始回合先做哈希契约核验；新候选(C3)在完全相同的task、init_state、model_seed、env_seed和init_state SHA上运行。新回合每片50/100，五片服务器runner连续接续且有完成标记；不重复跑历史控制，节省GPU。旧口径在这450回合是C0=368、C1=366、C2=388，headroom=20。原500口径411/408/431不能与受训练污染的init0–4混拼。

主指标为C3与C0/C1/C2的450配对成功率，报告Discordant flips、逐任务、新增失败/挽救、精确McNemar与episode bootstrap 95%CI、Recovery fraction仅在分母20成立时给出；同时报告adapter参数/字节、训练/推理时间、失败/异常回合。候选若不优于C0、或收益集中在单任务且伴随等量退步，则shared PEFT gate仍关闭；若C3显著优于C0并稳定跨任务，仍需新seed复现，才讨论P4互补，不直接实现Router。不得用离线MSE、事后成功标签或专家输出作路由输入。

本轮是历史/开发初态的受控配对实验，不是最终盲测。`offline_final_holdout`保持未触碰；纯W2是后续独立目标。额度约15%停止扩展归档本机/持久盘/Git并核SHA，约10%停止新GPU阶段、原生关当前107，留至少3%；所有关键备份、Git同步和107关机成功后才关闭本地Windows电脑。
