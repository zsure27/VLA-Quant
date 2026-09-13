"""双视觉W2组合的复现、范围检查与结果打包。"""
import argparse
import json
import math
import statistics
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def read(path):
    return json.loads(path.read_text())


def replay(old, new):
    a,b=[{r['sample']:r for r in read(p/'metrics.json')} for p in (old,new)]
    if set(a)!=set(b) or len(b)!=32:
        raise ValueError('复现样本不一致')
    for k in a:
        x,y=[r[k]['normalized_action']['mse'] for r in (a,b)]
        if not all(map(math.isfinite,(x,y))) or abs(x-y)>1e-8 or a[k]['raw_gripper_disagreement']!=b[k]['raw_gripper_disagreement']:
            raise ValueError('参照未复现：'+k)
    if read(old/'manifest.json')['profiles'] != read(new/'manifest.json')['profiles']:
        raise ValueError('参照profile不同')


def check_scope(scope, name):
    targets=scope['weight_targets']
    language=[n for n in targets if n.startswith('language_model.')]
    primary=[n for n in targets if n.startswith('vision_backbone.featurizer.')]
    fused=[n for n in targets if n.startswith('vision_backbone.fused_featurizer.')]
    expected_fused=0 if name=='primary-g64' else 105
    if len(targets)!=len(set(targets)) or (len(language),len(primary),len(fused),len(targets))!=(224,93,expected_fused,317+expected_fused):
        raise ValueError('量化范围错误')
    if scope['activation_targets'] or any(scope['weight_bits_by_target'][n]!=2 for n in targets):
        raise ValueError('要求W2A16')
    for n in targets:
        expected=64 if n in primary and name!='all-g128' else 128
        if scope['group_size_by_target'][n]!=expected:
            raise ValueError('分组路由错误：'+n)
    removed=scope['removed_clip_targets']
    if len(removed)!=160 or len(set(removed))!=160 or any(n not in language for n in removed) or scope['removed_visual_clip_targets']:
        raise ValueError('裁剪范围错误')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--group-run',type=Path,required=True)
    p.add_argument('--vision-run',type=Path,required=True)
    p.add_argument('--replay-only',action='store_true')
    args=p.parse_args()
    replay(args.group_run/'primary-g64',args.root/'primary-g64')
    replay(args.vision_run/'vision-w2-all',args.root/'all-g128')
    print('COMPOSITION_REFERENCES: PASS')
    if args.replay_only:
        return
    summary={}
    for name in ('all-g128','primary-g64','combined'):
        check_scope(read(args.root/name/'scope.json'),name)
        rows=read(args.root/name/'metrics.json')
        errors=[r['normalized_action']['mse'] for r in rows]
        if len(rows)!=32 or len({r['sample'] for r in rows})!=32 or not all(map(math.isfinite,errors)):
            raise ValueError('样本或误差无效')
        summary[name]=dict(mean_action_mse=statistics.mean(errors),median_action_mse=statistics.median(errors),
            max_action_mse=max(errors),gripper_disagreement_steps=round(sum(r['raw_gripper_disagreement']*8 for r in rows)),
            per_sample_action_mse={r['sample']:r['normalized_action']['mse'] for r in rows})
    b=summary['combined']['per_sample_action_mse']
    for name in ('all-g128','primary-g64'):
        a=summary[name]['per_sample_action_mse']
        if set(a)!=set(b):
            raise ValueError('配对样本不同')
        summary['combined_vs_'+name]=dict(improved=sum(b[k]<a[k] for k in a),worsened=sum(b[k]>a[k] for k in a))
    for filename,value in [('summary.json',summary),('complete.json',dict(status='COMPLETE',note='双视觉W2离线组合，不代表闭环成功率'))]:
        with (args.root/filename).open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False)
    output=Path(str(args.root)+'-review.zip')
    with ZipFile(output,'x',ZIP_DEFLATED) as z:
        for path in sorted(args.root.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.log'):
                z.write(path,str(path.relative_to(args.root)))
    print('AWQ_GROUP_COMPOSITION: COMPLETE')
    for name in ('all-g128','primary-g64','combined'):
        print(name,summary[name]['mean_action_mse'],summary[name]['gripper_disagreement_steps'],'/256')
    print('请下载并上传：',output)


if __name__=='__main__':
    main()
