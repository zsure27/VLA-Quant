"""Render audited, possibly partial same-backend results; never fill missing cases."""
from pathlib import Path
import argparse
import json
import statistics
import sys
import os

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'.local/report-libs'))
os.environ.setdefault('MPLCONFIGDIR',str(repo/'.local/matplotlib-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
d=json.loads((a.source/'summary-public.json').read_text())
curves=json.loads((a.source/'curves-public.json').read_text())
assert d['gate']=='PASS_AVAILABLE_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT'
a.output.mkdir(exist_ok=True,parents=True)
names={'bf16':'BF16 Torch','smoothquant_w4a8':'SQ W4A8','smoothquant_w4a4':'SQ W4A4','gptq_w4a16':'GPTQ W4A16'}
names={k:v for k,v in names.items() if k in d['coverage']}
palette={'smoothquant_w4a8':'#6556ae','smoothquant_w4a4':'#d6464d','gptq_w4a16':'#238a83'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,4,figsize=(16,9),sharex='col',gridspec_kw={'height_ratios':[3,3,1]},constrained_layout=True)
for col,s in enumerate(d['suites']):
    for row in [x for x in curves if x['suite']==s]:
        c=row['condition'];color=palette[c]
        for ai,field in enumerate(['position_mm','rotation_deg']):
            v=row[field];med=np.array([np.nan if x is None else x for x in v['median']]);lo=np.array([np.nan if x is None else x for x in v['p25']]);hi=np.array([np.nan if x is None else x for x in v['p75']])
            axes[ai,col].plot(row['step'],med,color=color,label=names[c],lw=1.6)
            axes[ai,col].fill_between(row['step'],lo,hi,color=color,alpha=.09)
        axes[2,col].step(row['step'],row['position_mm']['n'],color=color,where='post')
    for ai in range(3):axes[ai,col].grid(alpha=.17);axes[ai,col].set_ylim(bottom=0)
    axes[0,col].set_title(s)
    axes[2,col].set_xlabel('Executed control step')
    axes[0,col].set_ylabel('Position separation (mm)');axes[1,col].set_ylabel('SO(3) separation (deg)');axes[2,col].set_ylabel('Pairs N')
axes[0,0].legend(fontsize=8)
fig.suptitle('Independent PyTorch pi0.5 | initial5 | available same-input/noise pairs\nMedian/IQR only where N>=5; partial coverage is not a complete-method ranking'.replace('initial5','initial'+str(d['reset_indices'][0])))
fig.savefig(a.output/'pi05_available_drift_and_risk.png',dpi=160);plt.close(fig)
text=['# π0.5同后端量化轨迹接续：本轮实际结果','','## 综合结果与分析','',
      '当前统计仅为独立PyTorch后端、初态5的同40任务；原生JAX参考39/40与旧JAX→Torch等价检查失败均独立保留，不跨后端合并。', '',
      '|配置|完整病例/计划40|成功/完整病例|对应BF16成功|救回/新增失败|位置P95任务中位数 mm|旋转P95任务中位数 °|',
      '|---|---:|---:|---:|---:|---:|---:|']
for c,name in names.items():
    cov=d['coverage'][c];pairs=[x for x in d['pairs'] if x['condition']==c]
    if c=='bf16':values=[str(cov['success']),'-','0（自比较）','0（自比较）']
    elif pairs:values=[str(sum(x['reference_success'] for x in pairs)),str(sum(x['success'] and not x['reference_success'] for x in pairs))+'/'+str(sum(not x['success'] and x['reference_success'] for x in pairs)),f"{statistics.median(x['position_p95_mm'] for x in pairs):.3f}",f"{statistics.median(x['rotation_p95_deg'] for x in pairs):.3f}"]
    else:values=['-','-','-','-']
    text.append('|'+name+'|'+str(cov['complete'])+'/40|'+str(cov['success'])+'/'+str(cov['complete'])+'|'+'|'.join(values)+'|')
text+=['',f"共{d['complete_unique']}条唯一完整、{d['paired_quantized']}对量化/BF16配对。完整数不足40的配置按实际覆盖列示，不以缺项补失败、不据不等分母排名。待完成列表保存在summary-public.json；五初态/十二配置仍未全部完成。",'',
       '几何口径：同实际控制步的末端位置欧氏距离（mm）及SO(3)相对旋转主角（度），只比较共同可观测前缀；先计算每条轨迹P95，再取已有任务中位数。该分离包含闭环反馈，不是纯同状态量化误差，也不自动构成共性拐点或OPD补偿收益证据。', '',
       '![多suite位姿分离与风险集](pi05_available_drift_and_risk.png)','',
       '中位数/IQR只在至少5个任务有共同前缀时显示；风险集为实际N。稀疏suite不画聚合曲线，标量病例仍保留。', '',
       '## 数值门禁与控制问题','',
       '原始GPTQ414层产物完成后，动作保存重载门禁失败。源代码和CPU状态转换显示，旧生产器在量化后恢复了模型加载前的highest设置，而第二次原生模型构造使用high。独立GPU验证复用了同一414份权重（无重新量化），统一原生high后四个标定样本重复及重载逐位一致；反切旧TF32设置时四样本最大动作差约0.0010-0.0019。这支持精度状态合同不一致的解释，修复的是评测/持久化桥接，不是改进GPTQ量化公式或放宽误差容限。', '',
       'GPTQ框架名称未登记和源快照路径越界的早期拒绝均发生在正式轨迹前，保留为控制失败，不计模型成绩。额度观测递送中断触发的收尾、首次归档因可变状态文件而失败也单独保留；用户明确授权后，仅在静态归档逐成员验证、GPU闲和新鲜额度有效时恢复。静态归档不含414份参数分片，参数保留服务器及SHA索引。', '',
       '## 数据与成本','',
       '每例原始query、query-index固定噪声、实际执行前5动作、连续位姿、成功/终止信号与视频都重新按对应冻结审计器核验。只有PASS且完整的初态5开发轨迹进入统计，校准reset0及中断片不加分母。流程保持原生10×32流动作、模型seed7、每query的NumPy default_rng(0)噪声。','']
for c,name in names.items():
    rows=[x for x in d['records'] if x['condition']==c]
    text.append(f"- {name}：{sum(x['policy_queries'] for x in rows)}次策略查询；环境与query墙时合计{sum(x.get('wall_seconds_environment_and_queries') or 0 for x in rows):.3f}s；含校验/加载墙时合计{sum(x.get('wall_seconds_including_verification_load') or 0 for x in rows):.3f}s。")
text+=['','计时含环境、保存/视频、校验和加载；不是纯前向延迟或真实INT4内核速度。大参数只留服务器，原始增量与完整控制记录按私有备份规范另行封存。','',
       '## OpenVLA-OFT报告交付','',
       '另行交付的OpenVLA-OFT报告使用最新初态25的40任务，综合图剔除SQ W4A4，包含关键表、四suite漂移曲线、3D/XY/XZ/YZ固定样例，以及可切换40任务的离线交互HTML。PDF6页逐页渲染、HTML40任务切换/画布测试通过；280条参考/候选几何来自已验证归档，重算P95与原分析一致。输出保存在用户指定D盘目录。','']
text+=['','用户已取消SQ W4A4在初态25/35/45尚未运行的120例；已完成初态5/15的0/80与原始失败数据独立保留。取消项不作为待补病例或缺失失败。']
text=[x.replace('初态5','初态'+str(d['reset_indices'][0])) for x in text]
(a.output/'README_CN.md').write_text('\n'.join(text),encoding='utf-8')
for name in ['summary-public.json','curves-public.json']:
    (a.output/name).write_bytes((a.source/name).read_bytes())
print('PI05_AVAILABLE_REPORT_RENDERED',d['complete_unique'])
