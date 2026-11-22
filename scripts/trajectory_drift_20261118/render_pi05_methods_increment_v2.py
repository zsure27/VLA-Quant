"""Render only audited aggregate statistics; partial coverage stays explicit."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / '.local/report-libs'))
os.environ.setdefault('MPLCONFIGDIR', str(repo / '.local/matplotlib-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
summary = json.loads((a.source / 'summary-public.json').read_text())
curves = json.loads((a.source / 'curves-public.json').read_text())
assert summary['gate'] == 'PASS_AVAILABLE_PI05_SAME_BACKEND_RAW_GEOMETRY_AUDIT'
assert summary['reset_indices'] == [5]
names = {'bf16': 'BF16', 'smoothquant_w4a8': 'SmoothQuant W4A8',
         'gptq_w4a16': 'GPTQ W4A16', 'awq_w4a16': 'AWQ W4A16', 'omniquant_w4a16': 'OmniQuant LWC W4A16'}
colors = {'smoothquant_w4a8': '#6556ae', 'gptq_w4a16': '#238a83', 'awq_w4a16': '#df8c26', 'omniquant_w4a16': '#3b7fb5'}
a.output.mkdir(parents=True, exist_ok=False)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex='col', constrained_layout=True,
                         gridspec_kw={'height_ratios': [3, 3, 1]})
for column, suite in enumerate(summary['suites']):
    for row in curves:
        condition = row['condition']
        if row['suite'] != suite or condition not in colors:
            continue
        color = colors[condition]
        for i, field in enumerate(['position_mm', 'rotation_deg']):
            values = row[field]
            array = lambda key: np.asarray([np.nan if x is None else x for x in values[key]])
            axes[i, column].plot(row['step'], array('median'), color=color, label=names[condition], lw=1.6)
            axes[i, column].fill_between(row['step'], array('p25'), array('p75'), color=color, alpha=.1)
        axes[2, column].step(row['step'], row['position_mm']['n'], where='post', color=color)
    axes[0, column].set_title(suite)
    for row in range(3):
        axes[row, column].grid(alpha=.18)
        axes[row, column].set_ylim(bottom=0)
    axes[2, column].set_xlabel('Executed control step')
axes[0, 0].set_ylabel('Position separation (mm)')
axes[1, 0].set_ylabel('SO(3) separation (deg)')
axes[2, 0].set_ylabel('Observed pairs N')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Independent PyTorch pi0.5 | initial 5 | paired with same-backend BF16\n'
             'Median/IQR only for N>=5; unequal coverage is not a method ranking')
fig.savefig(a.output / 'pi05_awq_available_drift.png', dpi=160)
plt.close(fig)
lines = ['# π0.5多量化方法：本轮已审计结果', '',
         '## 结果与分析', '',
         '仅统计初态5、独立PyTorch同后端的实际完整病例；四suite各10任务。中断前缀不计失败、不补造缺失结果。', '',
         '|配置|完整/计划40|成功/完整|与BF16配对数|位置P95中位数 mm|旋转P95中位数 °|',
         '|---|---:|---:|---:|---:|---:|']
for condition, name in names.items():
    cov = summary['coverage'][condition]
    pairs = [x for x in summary['pairs'] if x['condition'] == condition]
    pos = f"{statistics.median(x['position_p95_mm'] for x in pairs):.3f}" if pairs else '—'
    rot = f"{statistics.median(x['rotation_p95_deg'] for x in pairs):.3f}" if pairs else '—'
    lines.append(f"|{name}|{cov['complete']}/40|{cov['success']}/{cov['complete']}|{len(pairs)}|{pos}|{rot}|")
lines += ['', 'AWQ各suite实际覆盖：', '']
for suite, value in summary['coverage']['awq_w4a16']['suites'].items():
    lines.append(f"- {suite}：{value['complete']}/10完整，{value['success']}成功。")
lines += ['', '![四suite同控制步的位姿分离和实际配对数](pi05_awq_available_drift.png)', '',
          '上述曲线是各自闭环轨迹在相同已执行控制步上的末端位置距离与SO(3)角距离，包含反馈和进度差异，不能直接称为同状态纯量化误差。', '',
          '成功率接近不表示动作或轨迹逐位一致；几何分离存在也不自动证明失败、统一拐点或OPD有效。需要进一步检查跨任务重复出现的偏离阶段，并区分同观测动作误差与闭环状态分离。', '',
          '当前覆盖不等时，表中中位数只描述已有病例，不据此排名方法。曲线仅在风险集N≥5时画中位数/IQR，较晚阶段的样本减少可能改变曲线形状。', '',
          'SQ W4A4历史数据仍在原始汇总中保留；用户已取消其初态25/35/45未完成部分。为便于观察其余方法，本图不叠加该失效配置；不将取消项补作失败。', '',
          '## 方法与证据边界', '',
          'AWQ为原始算法的W4A16静态量化：63块/414目标投影、32个校准查询；未微调模型、未根据闭环成绩调参数。完整模型冻结、finite、教师恢复和独立保存重载实际门禁通过后才采集轨迹。四suite首例属于正式40例，后续不重复计数。', '',
          '本实验使用独立PyTorch后端；原生JAX与历史JAX→Torch桥接失败单独保留，不跨后端合并。量化权重在当前路径反量化后执行，不把墙时或显存解释为真实INT4内核收益。', '',
          'OmniQuant为LWC原生单元重构适配（126单元/414投影），不含LET。仅量化裁剪参数参与优化，基座冻结；完整GPU保存重载门禁后才入场。不是完整论文复现。', '', '## 查询与成本', '']
for condition, name in names.items():
    rows = [x for x in summary['records'] if x['condition'] == condition]
    lines.append(f"- {name}：{sum(x['policy_queries'] for x in rows)}次策略查询；"
                 f"含验证/加载墙时合计{sum(x.get('wall_seconds_including_verification_load') or 0 for x in rows):.3f}s。")
lines += ['', '计时含加载、校验、环境、query及保存/视频。校准成本和未完成控制阶段在轮次日志独立记录，不作为上述完整轨迹的成本。']
(a.output / 'README_CN.md').write_text('\n'.join(lines), encoding='utf-8')
for name in ['summary-public.json', 'curves-public.json']:
    (a.output / name).write_bytes((a.source / name).read_bytes())
print('AWQ_AGGREGATE_REPORT_RENDERED', summary['coverage']['awq_w4a16']['complete'])
