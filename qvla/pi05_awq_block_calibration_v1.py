"""Memory-bounded native-block AWQ calibration, awaiting full-policy admission.

Uses pinned upstream 20-point scale search and clipping. A block is restored
bitwise before returning, so later blocks must capture the original teacher.
The caller supplies fixed-role calibration calls and exact native linear inputs.
"""
import gc
import torch
from torch import nn
from qvla.pi05_awq_replay_batch_v1 import ReplayBatch,native_scale_groups
from qvla.pi05_awq_native_components_v2 import original_functions,fold_norm_and_fcs,folded_activation,original_clip_with_output_padding
from qvla.pi05_packed_weight_v1 import pack,unpack,weight_sha


def registered_group(width):
    if width%128==0:
        return 128
    if width==4304:
        return 16
    raise ValueError('Input width outside registered 128/4304-to-16 groups')


def unique_tensor_bytes(tree):
    seen=set();size=0
    def visit(x):
        nonlocal size
        if isinstance(x,torch.Tensor):
            storage=x.untyped_storage();key=(str(x.device),storage.data_ptr())
            if key not in seen:
                seen.add(key);size+=storage.nbytes()
        elif isinstance(x,dict):
            for value in x.values():visit(value)
        elif isinstance(x,(list,tuple)):
            for value in x:visit(value)
    visit(tree)
    return size


@torch.no_grad()
def apply_scales(block,records,original_source):
    for row in records:
        targets=[block.get_submodule(name) for name in row['targets']]
        previous=block.get_submodule(row['previous'])
        original=original_functions(original_source,registered_group(targets[0].in_features))
        scale=row['scales'].to(targets[0].weight.device)
        if row['fold']=='norm':
            fold_norm_and_fcs(previous,targets,scale,original)
        elif row['fold']=='linear_pair':
            if len(targets)!=1 or previous.out_features!=targets[0].in_features:
                raise ValueError('Unverified linear pair')
            original['scale_fc_pair'](previous,targets[0],scale)
        elif row['fold']=='activation_pair':
            if len(targets)!=1:raise ValueError('Ambiguous activation pair')
            parent,name=row['previous'].rsplit('.',1)
            setattr(block.get_submodule(parent),name,folded_activation(previous,targets[0],scale,original))
        else:
            raise ValueError('Unknown AWQ fold')


@torch.no_grad()
def calibrate_block(block,calls,features,original_source,max_cpu_bytes=2*1024**3):
    if block.training or any(p.requires_grad for p in block.parameters()):
        raise ValueError('Frozen eval teacher required')
    groups=native_scale_groups(block)
    linears={n:m for n,m in block.named_modules() if isinstance(m,nn.Linear) and not n.endswith('.dense')}
    expected={n for g in groups for n in g['targets']}
    # GQA's v->o scaling is skipped by upstream, but o is still quantized.
    expected.add('self_attn.o_proj' if type(block).__name__=='GemmaDecoderLayer' else 'self_attn.out_proj')
    if set(linears)!=expected or set(features)!=expected:
        raise ValueError('Native projection/calibration scope differs')
    for name,linear in linears.items():
        x=features[name]
        if x.device.type!='cpu' or x.ndim!=2 or x.shape[1]!=linear.in_features or not torch.isfinite(x).all():
            raise ValueError('Malformed fixed calibration projection features')
        registered_group(linear.in_features)
    if unique_tensor_bytes([calls,features])>max_cpu_bytes:
        raise ValueError('Per-block calibration input budget exceeded')
    state={n:t.detach().cpu().clone() for n,t in block.state_dict().items()}
    activations={g['previous']:block.get_submodule(g['previous']) for g in groups if g['fold']=='activation_pair'}
    records=[];packed={}
    try:
        for group in groups:
            targets=[block.get_submodule(n) for n in group['targets']]
            original=original_functions(original_source,registered_group(targets[0].in_features))
            scorer=block.get_submodule(group['score'])
            wrapper=ReplayBatch(scorer,calls[group['score']],max_cpu_bytes)
            scale=original['search'](wrapper,targets,wrapper.inputs())
            records.append(dict(group,scales=scale.detach().cpu()))
            del wrapper
        # Upstream first searches all groups on the original block, then folds
        # all scales and rescales cached inputs before independent row clipping.
        apply_scales(block,records,original_source)
        scaled={k:v.clone() for k,v in features.items()}
        for row in records:
            for name in row['targets']:
                scaled[name].div_(row['scales'].to(scaled[name].dtype).view(1,-1))
        for name,linear in linears.items():
            width=registered_group(linear.in_features);original=original_functions(original_source,width)
            weight=linear.weight.detach().clone()
            if not any(x in name for x in ['q_proj','k_proj']):
                maximum=original_clip_with_output_padding(original,weight,scaled[name],width)
                view=weight.reshape(weight.shape[0],-1,width)
                view.clamp_(-maximum.to(view.device),maximum.to(view.device))
            quantized,scales,zeros=original['quantize'](weight,4,True,width,False,True)
            packed[name]=pack(quantized,list(scales.unbind(1)),list(zeros.unbind(1)),width)
            del weight,quantized
        return dict(schema_version=1,method='awq',block_class=type(block).__name__,scale_records=records,
                    weights=packed,original_files=original['original_files'],fixed_calibration=True,
                    full_policy_admitted=False,targets=sorted(packed))
    finally:
        for path,module in activations.items():
            parent,name=path.rsplit('.',1);setattr(block.get_submodule(parent),name,module)
        block.load_state_dict(state,strict=True)
        if not all(torch.equal(value,block.state_dict()[name].cpu()) for name,value in state.items()):
            raise AssertionError('Teacher block was not restored bitwise')
        gc.collect()


@torch.no_grad()
def load_block(block,artifact,original_source):
    if artifact.get('schema_version')!=1 or artifact.get('method')!='awq' or artifact['block_class']!=type(block).__name__:
        raise ValueError('AWQ artifact identity mismatch')
    original=original_functions(original_source,128)
    if artifact['original_files']!=original['original_files']:
        raise ValueError('AWQ source identity mismatch')
    groups=native_scale_groups(block)
    if [{k:r[k] for k in ['previous','targets','score','fold']} for r in artifact['scale_records']]!=groups:
        raise ValueError('AWQ scale graph mismatch')
    expected={n for n,m in block.named_modules() if isinstance(m,nn.Linear) and not n.endswith('.dense')}
    if set(artifact['weights'])!=expected or set(artifact['targets'])!=expected:
        raise ValueError('AWQ packed projection coverage mismatch')
    apply_scales(block,artifact['scale_records'],original_source)
    for name,entry in artifact['weights'].items():
        linear=block.get_submodule(name)
        if tuple(entry['shape'])!=tuple(linear.weight.shape) or entry['group_size']!=registered_group(linear.in_features):
            raise ValueError('AWQ packed projection shape/group mismatch')
        value=unpack(entry,linear.weight.device,linear.weight.dtype)
        if weight_sha(value)!=entry['weight_sha256']:
            raise ValueError('AWQ packed weight digest mismatch')
        linear.weight.copy_(value)
    block.requires_grad_(False)
