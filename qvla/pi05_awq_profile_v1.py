"""Strict loading for original-AWQ native-block artifacts; no INT4 kernel claim."""
from pathlib import Path
import torch
from qvla.infra.common import digest
from qvla.pi05_torch_backend_v2 import target_names,BACKEND_ID
from qvla.pi05_awq_block_calibration_v1 import load_block,apply_scales,native_scale_groups
from qvla.pi05_awq_native_components_v2 import ORIGINAL_FILES


@torch.no_grad()
def apply_profile(model,manifest,folder,original_source,scales_only=False,allow_pending=False):
    folder=Path(folder).resolve()
    if manifest.get('backend')!=BACKEND_ID or manifest.get('method')!='awq_w4a16_native_blocks_v1':
        raise ValueError('AWQ profile backend/method identity mismatch')
    if not allow_pending and (manifest.get('full_policy_admitted') is not True or manifest.get('gate')!='PASS_PI05_AWQ_NATIVE_414_RELOAD_FINITE'):
        raise ValueError('Full-policy AWQ admission is required before use')
    targets=set(target_names(model));entries=manifest['entries']
    if len(targets)!=414 or len(entries)!=63 or len({r['name'] for r in entries})!=63:
        raise ValueError('AWQ native model block/target scope differs')
    covered=set()
    for row in entries:
        path=(folder/row['file']).resolve()
        if not path.is_relative_to(folder) or path.is_symlink() or digest(path)!=row['sha256']:
            raise ValueError('AWQ block artifact path/digest mismatch')
        artifact=torch.load(path,map_location='cpu',weights_only=True)
        if artifact.get('original_files')!=ORIGINAL_FILES or artifact.get('method')!='awq' or artifact.get('schema_version')!=1:
            raise ValueError('AWQ block source/method identity mismatch')
        local={row['name']+'.'+n for n in artifact['weights']}
        if local&covered or not local<=targets:
            raise ValueError('AWQ packed projection coverage mismatch')
        block=model.get_submodule(row['name'])
        if artifact['block_class']!=type(block).__name__ or [{k:r[k] for k in ['previous','targets','score','fold']} for r in artifact['scale_records']]!=native_scale_groups(block):
            raise ValueError('AWQ native scale graph mismatch')
        if scales_only:
            apply_scales(block,artifact['scale_records'],original_source)
        else:
            load_block(block,artifact,original_source)
        covered.update(local)
    if covered!=targets:raise ValueError('Incomplete AWQ profile')
    if any(p.requires_grad for p in model.parameters()) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('AWQ full model must remain frozen and finite')
