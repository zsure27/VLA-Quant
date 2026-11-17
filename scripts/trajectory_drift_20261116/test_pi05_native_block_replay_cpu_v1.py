"""Tiny actual native Gemma blocks, not full-policy or AWQ admission."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--module',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import GemmaConfig
from transformers.models.gemma import modeling_gemma as native
spec=importlib.util.spec_from_file_location('independent_replay',a.module)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1);torch.manual_seed(7)
assert not torch.cuda.is_available()
a.output.mkdir(parents=True,exist_ok=False)
checks=[]
for adaptive in [False,True]:
    cfg=GemmaConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=8,
                    num_key_value_heads=1,head_dim=8,vocab_size=128,use_adarms=adaptive,
                    adarms_cond_dim=64 if adaptive else None,hidden_activation='gelu_pytorch_tanh')
    cfg._attn_implementation='eager'
    layer=native.GemmaDecoderLayer(cfg,layer_idx=1).eval()
    rope=native.GemmaRotaryEmbedding(config=cfg)
    for cached in [False,True]:
        length=3 if cached else 5;prefix=7 if cached else 0
        x=torch.randn(1,length,64);positions=torch.arange(prefix,prefix+length)[None]
        pair=(torch.randn(1,1,prefix,8),torch.randn(1,1,prefix,8)) if cached else None
        cache=m.LayerKVCache(1,pair)
        kwargs=dict(position_ids=positions,attention_mask=torch.zeros(1,1,length,length+prefix),
                    position_embeddings=rope(x,positions),past_key_value=cache,use_cache=not cached,
                    cache_position=positions[0],adarms_cond=torch.randn(1,64) if adaptive else None)
        call=m.capture_call((),dict(hidden_states=x,**kwargs),1)
        with torch.no_grad():expected=layer(x,**kwargs)[0]
        one=m.replay_call(layer,call)[0];two=m.replay_call(layer,call)[0]
        assert torch.equal(expected,one) and torch.equal(one,two)
        assert (call['cache_pair'] is None) if not cached else call['cache_pair'][0].shape[2]==prefix
        path=a.output/f'block-call-adaptive{adaptive}-cached{cached}.pt'
        torch.save(call,path);restored=torch.load(path,map_location='cpu',weights_only=True)
        assert torch.equal(one,m.replay_call(layer,restored)[0])
        try:cache[0]
        except KeyError:pass
        else:raise AssertionError('Cross-layer cache read accepted')
        checks.append(dict(adaptive=adaptive,cached_prefix=cached,layer_index=1,
                           prefix_tokens=prefix,native_output_bitwise=True,repeated_replay_bitwise=True,
                           serialized_replay_bitwise=True,cross_layer_cache_rejected=True))
result=dict(gate='PASS_NATIVE_GEMMA_PREFIX_EXPERT_BLOCK_REPLAY_CPU',checks=checks,
            module_sha256=hashlib.sha256(a.module.read_bytes()).hexdigest(),
            native_block_sha256=hashlib.sha256(inspect.getsource(native.GemmaDecoderLayer).encode()).hexdigest(),
            native_attention_sha256=hashlib.sha256(inspect.getsource(native.GemmaAttention).encode()).hexdigest(),
            full_model_loaded=False,gpu_model_admitted=False,awq_algorithm_run=False,
            scope='Actual tiny native blocks and layer-specific fresh KV snapshots only')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
