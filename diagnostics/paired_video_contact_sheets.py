"""CPU-only paired rollout frames; relative video progress, not matched states."""
import argparse,csv,hashlib,json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--pairs-csv',type=Path,required=True)
p.add_argument('--task-id',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
except OSError:font=ImageFont.load_default()
manifest=[]
with a.pairs_csv.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
for row in rows:
    if int(row['task_id'])!=a.task_id:continue
    decoded=[];sources=[]
    for case in ('bf16','w4'):
        path=Path(row[case+'_video']);reader=imageio.get_reader(str(path))
        count=reader.count_frames();meta=reader.get_meta_data();fps=float(meta['fps'])
        indices=sorted(set([0,(count-1)//3,2*(count-1)//3,count-1]))
        frames=[(i,Image.fromarray(reader.get_data(i))) for i in indices];reader.close()
        decoded.append((case,frames))
        sources.append(dict(case=case,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fps=fps,frame_count=count,selected_frame_indices=indices,selected_times_seconds=[i/fps for i in indices]))
    width,height=decoded[0][1][0][1].size;cols=max(len(fs) for _,fs in decoded)
    sheet=Image.new('RGB',(cols*width,60+2*(height+44)), 'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,5),'Task %s, init %s: BF16 vs W4'%(row['task_id'],row['init_state_index']),fill='black',font=font)
    draw.text((8,27),'Relative progress per video; trajectories/times are NOT matched states',fill='black',font=font)
    for n,(case,frames) in enumerate(decoded):
        y=60+n*(height+44);draw.text((8,y),case.upper()+' success='+row[case+'_success'],fill='black',font=font)
        for col,(index,frame) in enumerate(frames):
            sheet.paste(frame,(col*width,y+22))
            draw.text((col*width+4,y+22+height),'t=%.2fs'%(index/sources[n]['fps']),fill='black',font=font)
    name='task-%s-init-%02d.png'%(row['task_id'],int(row['init_state_index']));sheet.save(a.output/name)
    manifest.append(dict(task_id=int(row['task_id']),init_state_index=int(row['init_state_index']),
        source_shard=row['source_shard'],bf16_success=row['bf16_success'],w4_success=row['w4_success'],
        figure=name,sources=sources,note='Actual decoded rollout frames; differing closed-loop trajectories, not a causal error decomposition.'))
(a.output/'contact_sheet_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(output=str(a.output),figures=len(manifest))))
