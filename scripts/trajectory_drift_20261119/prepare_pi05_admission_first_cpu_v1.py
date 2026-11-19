"""Bounded stdlib-only readiness inventory. Never launches a model or installs packages."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inventory(root,relative):
    if relative is None:return dict(status='SOURCE_NOT_SPECIFIED_NOT_ADMITTED')
    path=root/relative
    if not path.is_dir():return dict(status='SOURCE_DIRECTORY_MISSING',relative_path=relative)
    files=[]
    # Source-only, bounded, no model pickle/weights and no source execution.
    total=0
    for item in sorted(path.rglob('*')):
        if item.is_symlink() or not item.is_file() or '.git' in item.parts or '__pycache__' in item.parts:continue
        if item.suffix.lower() not in {'.py','.cpp','.cc','.c','.h','.hpp','.cu','.cuh','.md','.toml'}:continue
        size=item.stat().st_size
        if size>8*1024**2:continue
        total+=size
        if total>64*1024**2:raise ValueError('Source inventory exceeds explicit 64MiB read budget')
        files.append(dict(path=str(item.relative_to(root)),bytes=size,sha256=digest(item)))
    return dict(status='SOURCE_BYTES_INDEXED_NOT_ADMITTED',relative_path=relative,files=files,
                indexed_files=len(files),indexed_bytes=total,upstream_revision_verified=False)


def main():
    p=argparse.ArgumentParser()
    for name in ['root','code','manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    assert manifest['executable'] is False and manifest['priority']=='all_configuration_admission_before_full_trajectory_expansion'
    root,code=a.root.resolve(),a.code.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    source_cache={};rows=[]
    for method in manifest['conditions']:
        source=method['source_root']
        if source not in source_cache:source_cache[source]=inventory(root,source)
        rows.append(dict(condition=method['id'],historical_status=method['status'],
            source_inventory=source_cache[source],cpu_work_is_not_gpu_admission=True,
            next_action=method.get('remaining','Reuse immutable verified profiles and complete-case identity index')))
    code_files=[]
    for relative in ['qvla/pi05_torch_backend_v2.py','qvla/execute_pi05_torch_episode_v2.py',
                     'qvla/pi05_smoothquant_v2.py','qvla/pi05_gptq_v1.py','qvla/pi05_omni_lwc_units_v1.py']:
        path=code/relative
        if path.exists():
            ast.parse(path.read_text(encoding='utf-8'))
            code_files.append(dict(path=relative,sha256=digest(path),syntax_only_no_import=True))
        else:code_files.append(dict(path=relative,missing=True))
    result=dict(gate='PASS_CPU_ADMISSION_FIRST_SOURCE_INVENTORY_ONLY',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=digest(a.manifest),disk_free_bytes=shutil.disk_usage(root).free,
        model_loaded=False,gpu_launched=False,source_directories=source_cache,conditions=rows,
        runtime_sources=code_files,all_configurations_ready_for_gpu=False,
        next_gpu_phase='Admission first; exact executable adapters and numerical gates still required')
    # JSON object keys cannot be null; use a list for the deduplicated inventory.
    result['source_directories']=list(source_cache.values())
    (a.output/'readiness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'plan.json').write_bytes(a.manifest.read_bytes())
    print(json.dumps({k:result[k] for k in ['gate','model_loaded','gpu_launched','all_configurations_ready_for_gpu','disk_free_bytes']}))


if __name__=='__main__':main()
