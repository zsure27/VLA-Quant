"""Pi05 graph binding for the pinned original DuQuant quantizer operations.

Each projection uses paired input/weight rotations. Frozen BF16 base weights
are streamed from CPU once and the resulting BF16 execution weight is cached.
The source quantizer arithmetic is untouched; this is not an INT4 kernel.
"""
import contextlib,hashlib,importlib,sys
from pathlib import Path
import torch
from qvla.pi05_torch_backend_v2 import target_names


def weight_sha(weight):
    return hashlib.sha256(weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def original_quantizer(source,source_sha):
    source=Path(source)
    for rel,expected in source_sha.items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest()!=expected:raise ValueError('DuQuant source changed')
    sys.path.insert(0,str(source))
    m=importlib.import_module('quantize.quantizer')
    if Path(m.__file__).resolve()!=source/'quantize/quantizer.py':raise ValueError('DuQuant import namespace collision')
    return m.UniformAffineQuantizer


@contextlib.contextmanager
def exact_rotation_matmul():
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:yield
    finally:torch.backends.cuda.matmul.allow_tf32=previous


class NativeProjection(torch.nn.Module):
    def __init__(self,base,quantizer,activation_bits,record=None):
        super().__init__()
        self.in_features,self.out_features=base.in_features,base.out_features
        self.native_dtype=base.weight.dtype
        self.register_buffer('weight',base.weight.detach().cpu().clone())
        self.register_buffer('bias',None if base.bias is None else base.bias.detach().cpu().clone())
        self.base_sha256=weight_sha(self.weight)
        self.block=128 if self.in_features%128==0 else 16
        if self.in_features%self.block:raise ValueError('Unsupported DuQuant projection width')
        common=dict(quant_method='duquant',block_size=self.block,max_rotation_step=1024,permutation_times=1,rotate=True)
        self.act=quantizer(n_bits=activation_bits,dynamic_method='per_token',**common)
        self.wq=quantizer(n_bits=4,dynamic_method='per_channel',group_size=self.block,shape=self.weight.shape,lwc=False,**common)
        self.activation_bits=activation_bits;self.quantized=True;self.cached_weight=None;self.cached_bias=None
        if record is not None:
            if record['base_sha256']!=self.base_sha256 or record['shape']!=list(self.weight.shape):raise ValueError('Wrong DuQuant checkpoint/projection')
            self.act.R=record['R'].cuda();self.act.permutation_list=None if record['permutation_list'] is None else record['permutation_list'].cuda()
            self.act.init_duquant_params=torch.tensor(1)
            self.wq.copy_duquant_params(self.act)

    def mode(self,quantized):
        self.quantized=quantized;self.act.change_n_bits(self.activation_bits if quantized else 16)
        self.wq.change_n_bits(4 if quantized else 16);self.cached_weight=None

    @torch.no_grad()
    def forward(self,value):
        shape=value.shape;flat=value.reshape(-1,self.in_features)
        with exact_rotation_matmul():
            x=self.act(flat.float().clone())
            if self.cached_weight is None:
                self.wq.copy_duquant_params(self.act)
                self.cached_weight=self.wq(self.weight.to(device=value.device,dtype=torch.float32)).to(self.native_dtype)
                self.cached_bias=None if self.bias is None else self.bias.to(device=value.device,dtype=self.native_dtype)
        result=torch.nn.functional.linear(x.to(self.native_dtype),self.cached_weight,self.cached_bias)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite DuQuant projection output')
        return result.reshape(*shape[:-1],self.out_features)

    def record(self):
        if not bool(self.act.init_duquant_params):raise ValueError('Projection was never calibrated')
        if weight_sha(self.weight)!=self.base_sha256:raise ValueError('Frozen base changed')
        return dict(shape=list(self.weight.shape),base_sha256=self.base_sha256,block=self.block,
                    R=self.act.R.detach().cpu().clone(),permutation_list=None if self.act.permutation_list is None else self.act.permutation_list.detach().cpu().clone())


def attach(model,quantizer,activation_bits,payload=None):
    names=target_names(model)
    if len(names)!=414:raise ValueError('Expected 414 native projections')
    if payload is not None and (payload['format']!='PI05_ORIGINAL_DUQUANT_NATIVE_V1' or set(payload['projections'])!=set(names) or payload['activation_bits']!=activation_bits):raise ValueError('Wrong DuQuant profile/scope')
    modules={}
    for name in names:
        parent,key=name.rsplit('.',1);container=model.get_submodule(parent);base=getattr(container,key)
        wrapper=NativeProjection(base,quantizer,activation_bits,None if payload is None else payload['projections'][name])
        setattr(container,key,wrapper);modules[name]=wrapper
    return modules


def set_mode(modules,quantized):
    for layer in modules.values():layer.mode(quantized)
