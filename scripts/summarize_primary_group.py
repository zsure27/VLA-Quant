"""检查主视觉分组单因素范围，打包离线诊断；不含权重文件。"""
import json
import math
import statistics
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(sys.argv[1])
    summary = {}
    for name, group in [('primary-g128', 128), ('primary-g64', 64)]:
        rows = json.loads((root/name/'metrics.json').read_text())
        scope = json.loads((root/name/'scope.json').read_text())
        targets = scope['weight_targets']
        if len(rows) != 32 or len({r['sample'] for r in rows}) != 32:
            raise ValueError('样本不完整')
        language = [n for n in targets if n.startswith('language_model.')]
        primary = [n for n in targets if n.startswith('vision_backbone.featurizer.')]
        if len(targets) != 317 or len(language) != 224 or len(primary) != 93:
            raise ValueError('目标范围错误')
        if any(scope['weight_bits_by_target'][n] != 2 for n in targets) or scope['activation_targets']:
            raise ValueError('不是W2A16')
        if any(scope['group_size_by_target'][n] != 128 for n in language) or any(scope['group_size_by_target'][n] != group for n in primary):
            raise ValueError('分组路由错误')
        if len(scope['removed_clip_targets']) != 160 or any(n not in language for n in scope['removed_clip_targets']) or scope['removed_visual_clip_targets']:
            raise ValueError('裁剪范围错误')
        errors = [r['normalized_action']['mse'] for r in rows]
        if not all(map(math.isfinite, errors)):
            raise ValueError('非有限误差')
        summary[name] = dict(mean_action_mse=statistics.mean(errors), median_action_mse=statistics.median(errors),
            max_action_mse=max(errors), gripper_disagreement_steps=round(sum(r['raw_gripper_disagreement']*8 for r in rows)),
            per_sample_action_mse={r['sample']:r['normalized_action']['mse'] for r in rows})
    a,b = [summary[n]['per_sample_action_mse'] for n in ('primary-g128','primary-g64')]
    if set(a) != set(b):
        raise ValueError('配对样本不同')
    summary['paired'] = dict(improved=sum(b[k]<a[k] for k in a), worsened=sum(b[k]>a[k] for k in a))
    for filename, value in [('summary.json',summary), ('complete.json',dict(status='COMPLETE',note='主视觉分组离线诊断，不代表闭环成功率'))]:
        with (root/filename).open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False)
    output = Path(str(root)+'-review.zip')
    with ZipFile(output,'x',ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log'):
                z.write(p,str(p.relative_to(root)))
    print('AWQ_PRIMARY_GROUP: COMPLETE')
    for n in ('primary-g128','primary-g64'):
        print(n,summary[n]['mean_action_mse'],summary[n]['gripper_disagreement_steps'],'/256')
    print('请下载并上传：',output)


if __name__ == '__main__':
    main()
