"""Measured initial25 report; private geometry input, no simulation or model calls."""
from pathlib import Path
import base64
import io
import json
import math
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / '.local/report-libs'), str(REPO)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from qvla.trajectory_diagnostics_v1 import rotation_difference

AREA = REPO / '.local/oft-report-20261116'
OUT = REPO / 'output/pdf/20261116_OpenVLA_OFT轨迹漂移报告'
OUT.mkdir(parents=True, exist_ok=False)
FIG = OUT / 'figures'
FIG.mkdir()
SUMMARY = REPO / 'Ex_Reports/_tasks/multisuite-trajectory-deviation-v1/rounds/20261115-061-oft-initial25-v1/summary-public.json'
data = json.loads(SUMMARY.read_text(encoding='utf-8'))
meta = json.loads((AREA / 'geometry-index-private.json').read_text(encoding='utf-8'))
assert meta['gate'] == 'PASS_280_SAME_INITIAL_INPUT_GEOMETRY_FROM_MEMBER_SHA'
geo = np.load(AREA / 'geometry.npz', allow_pickle=False)
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
LABEL = dict(libero_spatial='Spatial', libero_object='Object', libero_goal='Goal', libero_10='Long')
CN = dict(libero_spatial='空间关系', libero_object='物体变化', libero_goal='目标变化', libero_10='长程任务')
CONDS = ['bf16', 'awq_w4a16', 'gptq_w4a16', 'omniquant_w4a16', 'smoothquant_w4a8', 'qvla_w4a16']
NAMES = dict(zip(CONDS, ['BF16', 'AWQ W4A16', 'GPTQ W4A16', 'OmniQuant W4A16', 'SQ W4A8', 'QVLA mixed W4']))
COLORS = dict(zip(CONDS, ['#142232', '#2878b5', '#e58424', '#2d9668', '#9b59b6', '#009aaf']))
STYLES = dict(zip(CONDS, ['-', '-', '--', '-.', ':', '-']))
old = {'bf16': 75, 'awq_w4a16': 75, 'gptq_w4a16': 77, 'omniquant_w4a16': 76, 'smoothquant_w4a8': 75, 'qvla_w4a16': 75}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'savefig.facecolor':'white'})

def key(s, t, c):
    return f'{s}__task{t}__{c}'

def position(s, t, c, include_start=False):
    k = key(s,t,c)
    p = geo[k+'__position']
    if include_start:
        p = np.vstack([geo[k+'__start'], p])
    return p

errors = {}
for suite in SUITES:
    for task in range(10):
        ref = position(suite,task,'bf16')
        qr = geo[key(suite,task,'bf16')+'__quaternion']
        for condition in CONDS[1:]:
            candidate = position(suite,task,condition)
            qc = geo[key(suite,task,condition)+'__quaternion']
            n = min(len(ref),len(candidate))
            trans = np.linalg.norm(candidate[:n]-ref[:n],axis=1)*1000
            rot = np.array([np.linalg.norm(rotation_difference(a,b))*180/np.pi for a,b in zip(qc[:n],qr[:n])])
            row = next(x for x in data['metrics'] if x['suite']==suite and x['task_id']==task and x['condition']==condition)
            assert np.isclose(np.percentile(trans,95), row['translation_p95_mm'], atol=1e-7)
            assert np.isclose(np.percentile(rot,95), row['rotation_p95_deg'], atol=1e-7)
            errors[suite,task,condition] = (trans,rot)

# All-task curves: SQ4 is deliberately absent from every comparative graphic.
fig, axes = plt.subplots(3,4,figsize=(15.3,7.1),sharex='col',gridspec_kw={'height_ratios':[3,3,1.05]},constrained_layout=True)
for col,suite in enumerate(SUITES):
    valid_ends = []
    for c in CONDS[1:]:
        for ai in range(2):
            values = [errors[suite,t,c][ai] for t in range(10)]
            grid = np.full((10,max(map(len,values))),np.nan)
            for j,v in enumerate(values): grid[j,:len(v)] = v
            risk = np.isfinite(grid).sum(0)
            valid = risk>=5
            step = np.arange(1,grid.shape[1]+1)
            median = np.nanmedian(grid,axis=0)
            lower,upper = np.nanpercentile(grid,[25,75],axis=0)
            axes[ai,col].plot(step,np.where(valid,median,np.nan),color=COLORS[c],ls=STYLES[c],lw=1.6,label=NAMES[c])
            axes[ai,col].fill_between(step,lower,upper,where=valid,color=COLORS[c],alpha=.075)
            if ai==0:
                axes[2,col].step(step,risk,where='post',color=COLORS[c],lw=1.2,ls=STYLES[c])
                valid_ends.append(step[valid][-1])
    for ai in range(3):
        axes[ai,col].grid(alpha=.17)
        axes[ai,col].set_ylim(bottom=0)
        axes[ai,col].set_xlim(0,max(valid_ends)+2)
    axes[0,col].set_title(LABEL[suite]+' | 10 tasks',weight='bold')
    axes[0,col].axhline(5,color='#788390',lw=.8,ls='--')
    axes[1,col].axhline(2,color='#788390',lw=.8,ls='--')
    axes[2,col].set_yticks([0,5,10]);axes[2,col].set_xlabel('Executed control step')
    if col==0:
        axes[0,col].set_ylabel('Position separation (mm)')
        axes[1,col].set_ylabel('SO(3) separation (deg)')
        axes[2,col].set_ylabel('Pairs N')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False,fontsize=9)
fig.savefig(FIG/'01_综合漂移曲线_不含SQ4.png',dpi=180)
plt.close(fig)

def multiview(suite,task,path):
    fig = plt.figure(figsize=(14.2,8.1),layout='constrained')
    ax3=fig.add_subplot(2,2,1,projection='3d')
    projections=[fig.add_subplot(2,2,2),fig.add_subplot(2,2,3),fig.add_subplot(2,2,4)]
    origin=geo[key(suite,task,'bf16')+'__start']
    paths={c:(position(suite,task,c,True)-origin)*1000 for c in CONDS}
    union=np.concatenate(list(paths.values()));lo=union.min(0);hi=union.max(0);center=(lo+hi)/2;span=max(float(np.max(hi-lo))*1.1,20)
    for c,p in paths.items():
        label=NAMES[c]+f" ({len(p)-1} steps)"
        width=2.1 if c=='bf16' else 1.35
        ax3.plot(*p.T,color=COLORS[c],ls=STYLES[c],lw=width,label=label)
        ax3.scatter(*p[-1],s=22,color=COLORS[c],marker='o',depthshade=False)
        for ax,(a,b) in zip(projections,[(0,1),(0,2),(1,2)]):
            ax.plot(p[:,a],p[:,b],color=COLORS[c],ls=STYLES[c],lw=width)
            ax.scatter(p[-1,a],p[-1,b],s=22,color=COLORS[c],zorder=4)
    ax3.scatter(0,0,0,color='black',marker='D',s=23,depthshade=False)
    ax3.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)')
    ax3.set_xlim(center[0]-span/2,center[0]+span/2);ax3.set_ylim(center[1]-span/2,center[1]+span/2);ax3.set_zlim(center[2]-span/2,center[2]+span/2)
    ax3.set_box_aspect((1,1,1));ax3.view_init(elev=25,azim=-58);ax3.set_title('3D | shared initial end-effector position = origin',fontsize=10)
    for ax,(a,b),title in zip(projections,[(0,1),(0,2),(1,2)],['XY | top view','XZ | side view','YZ | front view']):
        ax.set(xlabel='XYZ'[a]+' (mm)',ylabel='XYZ'[b]+' (mm)',title=title)
        ax.set_xlim(center[a]-span/2,center[a]+span/2);ax.set_ylim(center[b]-span/2,center[b]+span/2)
        ax.scatter(0,0,color='black',marker='D',s=20,zorder=5);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    fig.legend(*ax3.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.savefig(path,dpi=170)
    plt.close(fig)

for suite in SUITES: multiview(suite,0,FIG/(LABEL[suite]+'_task0_多视角.png'))

pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('CJKBold','C:/Windows/Fonts/msyhbd.ttc'))
W,H=landscape(A4);M=34
pdf=canvas.Canvas(str(OUT/'OpenVLA-OFT_量化轨迹漂移报告_20261116.pdf'),pagesize=(W,H))
pdf.setTitle('OpenVLA-OFT 多视角量化轨迹漂移报告')
pdf.setAuthor('VLA实验记录')
normal=ParagraphStyle('normal',fontName='CJK',fontSize=9.2,leading=15,textColor=colors.HexColor('#243648'))
small=ParagraphStyle('small',parent=normal,fontSize=8,leading=12)
bold=ParagraphStyle('bold',parent=normal,fontName='CJKBold',fontSize=10.8,leading=16)

def para(text,x,y,width,style=normal):
    p=Paragraph(text,style);_,height=p.wrap(width,1000);p.drawOn(pdf,x,y-height);return y-height

def header(title,number,kicker='实测轨迹 / INITIAL 25'):
    pdf.setFillColor(colors.HexColor('#16314c'));pdf.rect(0,H-74,W,74,fill=1,stroke=0)
    pdf.setFillColor(colors.HexColor('#91cfdb'));pdf.setFont('CJK',8);pdf.drawString(M,H-23,kicker)
    pdf.setFillColor(colors.white);pdf.setFont('CJKBold',19);pdf.drawString(M,H-51,title)
    pdf.setFillColor(colors.HexColor('#65758a'));pdf.setFont('CJK',7.5)
    pdf.drawString(M,18,'OpenVLA-OFT · LIBERO四suite · 实测数据；无新模型推理 / 2026-11-16')
    pdf.drawRightString(W-M,18,f'{number} / 6')

header('量化后仍能成功，但轨迹并不完全相同',1,'OpenVLA-OFT / 轨迹漂移简报')
y=H-92
y=para('本轮：初态25，4 suite × 10任务；每个量化配置40条，BF16参考39/40成功。累计：初态5、15、25，各配置120条，共840条唯一完整轨迹。未完成五初态目标，未进行补偿训练。',M,y,W-2*M)
y-=12
table_data=[['配置','本轮成功 / 40','累计成功 / 120','位置P95中位数\nmm','旋转P95中位数\n度','救回 / 新增失败']]
table_data.append(['BF16 reference','39 / 40','114 / 120','0（自比较）','0（自比较）','-'])
for c in CONDS[1:]:
    rows=[x for x in data['metrics'] if x['condition']==c];v=data['coverage'][c]
    table_data.append([NAMES[c],f"{v['success']} / 40",f"{old[c]+v['success']} / 120",f"{statistics.median(x['translation_p95_mm'] for x in rows):.2f}",f"{statistics.median(x['rotation_p95_deg'] for x in rows):.2f}",f"{v['rescue']} / {v['breaks']}"])
table=Table(table_data,colWidths=[166,108,118,129,125,126],rowHeights=[34]+[25]*6)
table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'CJK'),('FONTNAME',(0,0),(-1,0),'CJKBold'),('FONTSIZE',(0,0),(-1,-1),8.5),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf3')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#16314c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')]),('ALIGN',(1,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#b7c8d7')),('LEFTPADDING',(0,0),(-1,-1),8)]))
_,th=table.wrap(W-2*M,400);table.drawOn(pdf,M,y-th);y-=th+10
y=para('定义：位置/旋转误差按同一实际控制步与配对BF16比较，仅使用双方共同可观测前缀；先求每条轨迹P95，再取40任务中位数。救回=BF16失败而量化成功；新增失败相反。表中误差统计仅对应本轮初态25。',M,y,W-2*M,small)
y-=13
y=para('SQ W4A4单独说明（综合图中全部剔除）',M,y,W-2*M,bold)
y=para('本轮0/40，三初态累计0/120；位置P95的任务中位数362.46 mm，旋转34.57°，本轮四suite均出现明显大幅偏离。相同权重位宽下，SQ W4A8为36/40，说明该已测试配方对A4激活精度十分敏感。但这不能证明所有W4A4方法均失效，也不能仅从曲线确定具体层、夹爪或接触机制。',M,y,W-2*M)
y-=11
y=para('简要结论',M,y,W-2*M,bold)
y=para('① 成功率不能代替轨迹一致性：其余配置仍有约30-68 mm的位置P95任务中位数。② 漂移出现的时间和方向因任务、方法而异，目前没有证据支持共享单一拐点。③ 后续应在同一状态下比较教师/量化动作，并关联动作阶段与接触事件，再检验局部补偿；本报告不把越界当作OPD有效的证据。',M,y,W-2*M)
assert y>30,(y,'summary page overflow')
pdf.showPage()

header('四suite综合漂移：比较全部40个任务',2)
pdf.drawImage(str(FIG/'01_综合漂移曲线_不含SQ4.png'),M,92,width=W-2*M,height=H-180,preserveAspectRatio=True,anchor='c')
para('实线/虚线：任务中位数；浅色带：25%-75%区间。曲线仅显示同时至少5个任务仍有配对观测的区间；底部N显示实际风险集。5 mm / 2°虚线沿用此前重复控制阈值，仅作描述，既不是安全边界，也不是失败判据。',M,75,W-2*M,small)
pdf.showPage()
for i,suite in enumerate(SUITES,3):
    header(f'{LABEL[suite]}：3D路径与三个正交投影',i,f'固定样例 / {CN[suite]} / task 0 / 初态25')
    pdf.drawImage(str(FIG/(LABEL[suite]+'_task0_多视角.png')),M,76,width=W-2*M,height=H-158,preserveAspectRatio=True,anchor='c')
    para('坐标为相对共同初始末端位置的世界坐标位移（mm）；只平移共同原点，不旋转、不做DTW或时间重采样。黑色菱形为共同起点，圆点为各自真实终点；图例给出实际控制步数。展示完整路径；上一页误差仅比较共同前缀。',M,64,W-2*M,small)
    pdf.showPage()
pdf.save()

# Offline interactive report. Payload contains only measured geometry, never images/actions.
viewer={}
for suite in SUITES:
    viewer[suite]={}
    for task in range(10):
        origin=geo[key(suite,task,'bf16')+'__start']
        viewer[suite][str(task)]={c:np.round((position(suite,task,c,True)-origin)*1000,2).tolist() for c in CONDS}

def image_data(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
rows_html=''.join('<tr>'+''.join('<th>'+str(x).replace('\n','<br>')+'</th>' for x in row)+'</tr>' if i==0 else '<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for i,row in enumerate(table_data))
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenVLA-OFT量化轨迹漂移报告</title><style>
*{box-sizing:border-box}body{font:15px/1.7 'Microsoft YaHei',sans-serif;margin:0;background:#eef3f7;color:#20364b}main{max-width:1240px;margin:auto;padding:32px}header{background:#16314c;color:white;padding:28px 34px;border-radius:18px}h1{font-size:30px;margin:4px 0}h2{font-size:23px;color:#173c58}section{background:white;padding:24px 28px;margin:22px 0;border-radius:16px;box-shadow:0 3px 14px #18344b0a}.sub{color:#63788a;font-size:13px}.warn{background:#fff2e9;border-left:4px solid #e19b55;padding:15px 20px;margin-top:20px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dfe7ee;text-align:center}th{background:#e6eef4}th:first-child,td:first-child{text-align:left}select,button{font:inherit;padding:6px 12px;border:1px solid #cbd8e3;border-radius:7px;background:white}.controls{display:flex;gap:13px;align-items:center;flex-wrap:wrap}.checks{display:flex;gap:15px;flex-wrap:wrap;margin:14px 0}.view{display:grid;grid-template-columns:1.25fr 1fr;gap:12px}canvas{width:100%;height:430px;background:#f9fbfd;border:1px solid #e2eaf0;border-radius:10px;touch-action:none}.smallviews{display:grid;grid-template-columns:1fr;gap:8px}.smallviews canvas{height:135px}input[type=range]{flex:1;min-width:220px}.img{max-width:100%;height:auto}details{margin:15px 0}footer{font-size:12px;color:#698094;padding:18px}@media(max-width:800px){main{padding:12px}.view{grid-template-columns:1fr}table{font-size:11px}th,td{padding:5px}section{padding:14px}}
</style><main><header><div>实测数据 · LIBERO四suite · 2026-11-16</div><h1>OpenVLA-OFT量化轨迹漂移报告</h1><div>综合图剔除SQ W4A4；支持40个任务的离线3D旋转、正交投影与逐步查看</div></header>
<section><h2>关键数据</h2><p>本轮初态25，每配置40条。累计为初态5、15、25，同40任务、840条唯一完整轨迹；不代表五初态或跨模型全部完成。</p><table>TABLE</table><p class="sub">位置/旋转：相对BF16、同一步共同前缀；每条轨迹P95再取40任务中位数（本轮初态25）。累计成功数跨三个相同协议初态；救回/新增失败仅本轮。</p><div class="warn"><b>SQ W4A4失效单列说明</b><br>本轮0/40，累计0/120；位置P95中位数362.46 mm，旋转34.57°，四suite均有大幅偏离。SQ W4A8同轮36/40，表明已测试配方对A4激活精度敏感。不能据此推论所有W4A4方法均失效，或直接确定具体层/夹爪/接触机制。该配置不进入以下任何综合图或交互轨迹。</div></section>
<section><h2>40任务综合漂移对比</h2><img class="img" src="OVERVIEW"><p class="sub">每suite10任务。中位数/IQR仅在N≥5的区间展示，下行N为仍有配对观测的任务数。阈值沿用此前重复控制，非安全/失败边界。这里测量的是闭环位姿分离，包含状态反馈放大。</p></section>
<section><h2>从3D与2D查看同一条轨迹</h2><p>拖动左图旋转；右侧同步显示XY、XZ、YZ。固定suite与task后所有方法采用同一初态25。图形坐标为相对共同起点的世界坐标位移，单位mm。</p><div class="controls"><label>Suite <select id="suite"></select></label><label>任务 <select id="task"></select></label><button id="reset">重置3D视角</button><button id="full">显示完整路径</button></div><div id="checks" class="checks"></div><div class="controls"><label>显示到控制步</label><input id="step" type="range" min="0" value="0"><span id="steptext"></span></div><div class="view"><canvas id="scene"></canvas><div class="smallviews"><canvas id="xy"></canvas><canvas id="xz"></canvas><canvas id="yz"></canvas></div></div><p id="lengths" class="sub"></p><p class="sub">黑色菱形：共同起点；圆点：当前显示步/实际终点。完整路径保留各自真实终止长度，不补齐、不旋转对齐、不进行DTW或时间归一化。仅可观察轨迹，不代表定位到因果拐点。</p></section>
<section><h2>固定样例：各suite task 0</h2>EXAMPLES</section><section><h2>结论与简要分析</h2><ol><li>除SQ W4A4外，多数配置仍保持较高成功率，但轨迹存在可见偏离，成功不等于按原路径运动。</li><li>当前曲线中漂移的出现时间、空间方向和变化形态依任务/方法而异，尚不能认定一个跨所有任务和方法的共享单一拐点。</li><li>下一步需要同状态交叉查询教师与量化动作，并标注阶段/夹爪/接触事件，才有条件检验局部补偿。未训练OPD，也未证明其收益。</li></ol><p class="sub">数据来源：已验证的40条BF16参考与240条量化原始轨迹；原始包及选用JSON成员SHA验证通过，首观测/初态与起点逐位配对，几何P95与既有最终指标复核一致。综合视图240条中仅展示200条非SQ4候选及40条BF16参考。3D案例图选择task0，未按成功率或漂移幅度挑例。</p></section><footer>本文件完全离线运行，无外部脚本、无模型推理、无服务器连接。OpenVLA-OFT，不与原始OpenVLA模型混称。</footer></main>
<script>const DATA=PAYLOAD,NAMES=LABELS,COLORS=PALETTE,CONDS=CONDITIONS;let az=-.95,el=.43,drag=null;const suite=document.getElementById('suite'),task=document.getElementById('task'),step=document.getElementById('step');for(const [s,n] of Object.entries({libero_spatial:'Spatial',libero_object:'Object',libero_goal:'Goal',libero_10:'Long'}))suite.add(new Option(n,s));for(let i=0;i<10;i++)task.add(new Option(i,i));const enabled=new Set(CONDS);for(const c of CONDS){const lab=document.createElement('label');lab.style.color=COLORS[c];const box=document.createElement('input');box.type='checkbox';box.checked=true;box.onchange=()=>{box.checked?enabled.add(c):enabled.delete(c);draw()};lab.append(box,document.createTextNode(' '+NAMES[c]));document.getElementById('checks').append(lab)}
function current(){return DATA[suite.value][task.value]}function changed(){let m=Math.max(...Object.values(current()).map(a=>a.length-1));step.max=m;step.value=m;draw()}suite.onchange=changed;task.onchange=changed;step.oninput=draw;document.getElementById('reset').onclick=()=>{az=-.95;el=.43;draw()};document.getElementById('full').onclick=()=>{step.value=step.max;draw()};const scene=document.getElementById('scene');scene.onpointerdown=e=>{scene.setPointerCapture(e.pointerId);drag=[e.clientX,e.clientY]};scene.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.009;el=Math.max(-1.45,Math.min(1.45,el+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];draw()};scene.onpointerup=()=>drag=null;
function setup(id){const c=document.getElementById(id),rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=rect.width*dpr;c.height=rect.height*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);return [ctx,rect.width,rect.height]}
function draw(){const paths=current(),limit=+step.value;document.getElementById('steptext').textContent=limit+' / '+step.max;document.getElementById('lengths').textContent=CONDS.map(c=>NAMES[c]+': '+(paths[c].length-1)+'步').join('　|　');const points=Object.values(paths).flat(),lo=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))),hi=[0,1,2].map(i=>Math.max(...points.map(p=>p[i]))),center=lo.map((x,i)=>(x+hi[i])/2),span=Math.max(...lo.map((x,i)=>hi[i]-x),20)*1.15;for(const [id,axes,title] of [['scene',null,'3D · 拖动旋转'],['xy',[0,1],'XY 俯视'],['xz',[0,2],'XZ 侧视'],['yz',[1,2],'YZ 正视']]){const [ctx,w,h]=setup(id);ctx.fillStyle='#243e56';ctx.font='13px Microsoft YaHei';ctx.fillText(title,12,22);let scale=Math.min(w-85,h-56)/span;
const project=p=>{if(axes)return [w/2+(p[axes[0]]-center[axes[0]])*scale,h/2-(p[axes[1]]-center[axes[1]])*scale+8];const q=p.map((x,i)=>x-center[i]),xx=q[0]*Math.cos(az)-q[1]*Math.sin(az),yy=q[0]*Math.sin(az)+q[1]*Math.cos(az),zz=yy*Math.sin(el)+q[2]*Math.cos(el);return [w/2+xx*scale*.82,h/2-zz*scale*.82+12]};ctx.strokeStyle='#cfdae4';ctx.lineWidth=1;for(let a=0;a<3;a++){if(axes&&!axes.includes(a))continue;let p=center.slice(),q=center.slice();p[a]-=span*.45;q[a]+=span*.45;const u=project(p),v=project(q);ctx.beginPath();ctx.moveTo(...u);ctx.lineTo(...v);ctx.stroke();ctx.fillStyle='#6b8295';ctx.fillText('XYZ'[a]+' (mm)',v[0]+3,v[1]+4)}
for(const c of CONDS){if(!enabled.has(c))continue;const p=paths[c].slice(0,limit+1);if(!p.length)continue;ctx.strokeStyle=COLORS[c];ctx.fillStyle=COLORS[c];ctx.lineWidth=c==='bf16'?2.2:1.6;ctx.setLineDash(c==='gptq_w4a16'?[6,3]:c==='smoothquant_w4a8'?[2,3]:[]);ctx.beginPath();p.forEach((v,i)=>{const x=project(v);i?ctx.lineTo(...x):ctx.moveTo(...x)});ctx.stroke();ctx.setLineDash([]);const last=project(p[p.length-1]);ctx.beginPath();ctx.arc(...last,3.5,0,Math.PI*2);ctx.fill()}const o=project([0,0,0]);ctx.fillStyle='#142232';ctx.beginPath();ctx.moveTo(o[0],o[1]-5);ctx.lineTo(o[0]+5,o[1]);ctx.lineTo(o[0],o[1]+5);ctx.lineTo(o[0]-5,o[1]);ctx.closePath();ctx.fill();ctx.font='11px Microsoft YaHei';ctx.fillStyle='#647b8e';ctx.fillText('共同起点 = 0；所有坐标单位mm',12,h-9)}}window.addEventListener('resize',draw);changed();</script></html>'''
html=html.replace('TABLE',rows_html).replace('OVERVIEW',image_data(FIG/'01_综合漂移曲线_不含SQ4.png'))
examples=''.join(f'<details><summary>{LABEL[s]} / task 0 / 初态25：3D、XY、XZ、YZ</summary><img class="img" src="{image_data(FIG/(LABEL[s]+"_task0_多视角.png"))}"></details>' for s in SUITES)
html=html.replace('EXAMPLES',examples).replace('PAYLOAD',json.dumps(viewer,separators=(',',':'))).replace('LABELS',json.dumps(NAMES)).replace('PALETTE',json.dumps(COLORS)).replace('CONDITIONS',json.dumps(CONDS))
(OUT/'OpenVLA-OFT_量化轨迹漂移报告_交互版.html').write_text(html,encoding='utf-8')
(OUT/'报告数据与口径.json').write_text(json.dumps(dict(model='OpenVLA-OFT',covered_suites=SUITES,tasks_per_suite=10,reset_index=25,geometry_episodes=280,interactive_comparative_conditions=CONDS,excluded_from_comparative_graphs='smoothquant_w4a4',coordinate_frame='World axes translated by the same BF16 initial end-effector position, millimetres',source_archives=meta['sources'],summary_sha256=__import__('hashlib').sha256(SUMMARY.read_bytes()).hexdigest(),pdf_pages=6),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),pdf_pages=6,geometry_verified_episodes=280,compared_non_sq4_candidates=200),ensure_ascii=False))
