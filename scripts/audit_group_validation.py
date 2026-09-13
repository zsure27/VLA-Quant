"""固定分组候选的既有独立验证集审计及打包。"""
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from check_group_composition import check_scope, read


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def preflight(base, group, validation):
    profiles={}
    for path,expected in [(base/'profiles/w2.pt','4fe1d2aa9a4e89fbaa5f9eb358ac6526d899195f774378b476742b801c7f4ccc'),
                          (group/'profiles/w2-g64.pt','947a849114a402978ae60995a652cd3212ded8d66f6ee7e0f3eeb3484712b3b8')]:
        if sha(path)!=expected:
            raise ValueError('固定profile变化：'+str(path))
        profiles[str(path)]=expected
    rows=read(validation/'samples/manifest.json')['samples']
    original=read(base/'samples/manifest.json')['samples']
    if len(rows)!=32 or len({(r['suite'],r['episode']) for r in rows})!=32:
        raise ValueError('验证轨迹数异常')
    if {(r['suite'],r['episode']) for r in rows}&{(r['suite'],r['episode']) for r in original}:
        raise ValueError('验证与旧校准/诊断轨迹重叠')
    samples={}
    for r in rows:
        if Path(r['file']).name!=r['file'] or r['suite']!='libero_spatial_no_noops':
            raise ValueError('验证范围错误')
        value=sha(validation/'samples'/r['file'])
        if value!=r['sha256']:
            raise ValueError('验证样本内容变化')
        samples[r['file']]=value
    if len(samples)!=32 or len(set(samples.values()))!=32 or set(samples.values())&{r['sha256'] for r in original}:
        raise ValueError('重复样本或内容重叠')
    actual={p.name for p in (validation/'samples').glob('sample-*.npz')}
    if actual!=set(samples) or read(validation/'teacher/manifest.json')['samples']!=samples:
        raise ValueError('验证目录/教师样本不一致')
    return dict(status='PASS',profiles=profiles,samples=samples,trajectories=32,
                note='复用曾用于语言量化验证的32条轨迹；不是从未查看过的最终测试集')


def main():
    base,group,validation,root=[Path(os.environ[k]) for k in ('BASE','GROUP_RUN','VALIDATION','OUT')]
    audit=preflight(base,group,validation)
    if '--preflight' in sys.argv:
        print(json.dumps(audit,ensure_ascii=False,indent=2))
        return
    repeat=read(root/'repeat/metrics.json')
    if len(repeat)!=32 or any(not math.isfinite(r['normalized_action']['mse']) or r['normalized_action']['mse']>1e-10 for r in repeat):
        raise ValueError('教师重复未通过')
    summary={}
    for name in ('all-g128','combined'):
        check_scope(read(root/name/'scope.json'),name)
        manifest=read(root/name/'manifest.json')
        if manifest['samples']!=audit['samples']:
            raise ValueError('推理样本不一致')
        rows=read(root/name/'metrics.json')
        errors=[r['normalized_action']['mse'] for r in rows]
        if len(rows)!=32 or {r['sample'] for r in rows}!=set(audit['samples']) or not all(map(math.isfinite,errors)):
            raise ValueError('指标不完整')
        summary[name]=dict(mean_action_mse=statistics.mean(errors),median_action_mse=statistics.median(errors),
            max_action_mse=max(errors),gripper_disagreement_steps=round(sum(r['raw_gripper_disagreement']*8 for r in rows)),
            per_sample_action_mse={r['sample']:r['normalized_action']['mse'] for r in rows})
    a,b=[summary[n]['per_sample_action_mse'] for n in ('all-g128','combined')]
    summary['paired']=dict(improved=sum(b[k]<a[k] for k in a),worsened=sum(b[k]>a[k] for k in a),unchanged=sum(b[k]==a[k] for k in a))
    for filename,value in [('summary.json',summary),('complete.json',dict(status='COMPLETE',note='固定候选的既有独立轨迹离线验证，非闭环成功率'))]:
        with (root/filename).open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False)
    output=Path(str(root)+'-review.zip')
    with ZipFile(output,'x',ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.log'):
                z.write(path,str(path.relative_to(root)))
    print('AWQ_GROUP_VALIDATION: COMPLETE')
    for n in ('all-g128','combined'):
        print(n,summary[n]['mean_action_mse'],summary[n]['gripper_disagreement_steps'],'/256')
    print('请下载并上传：',output)


if __name__=='__main__':
    main()
