"""DuQuant decoder API/dtype shim for CPU contracts, not full-model admission.

Original quantizers, LET/LWC and transformation math remain untouched. Only the
Transformers4.40 RoPE call and an unconditional FP16 cast are adapted. Cache mode
is deliberately rejected until separately tested; current source is never edited.
"""
from pathlib import Path
import hashlib,sys,types
PIN='ba3715401c12fd5e09e853a0837d83552b982b98807633ea01db4a6e1fa02f43'

def load_decoder_class(source):
 source=Path(source);file=source/'models/int_llama_layer.py'
 if hashlib.sha256(file.read_bytes()).hexdigest()!=PIN:raise ValueError('DuQuant decoder source changed')
 text=file.read_text()
 old='cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)'
 if text.count(old)!=1:raise ValueError('Unexpected RoPE call')
 text=text.replace(old,'cos, sin = self.rotary_emb(value_states, position_ids)')
 old='hidden_states = self.post_attention_layernorm(hidden_states).half()'
 if text.count(old)!=1:raise ValueError('Unexpected forced dtype')
 text=text.replace(old,'hidden_states = self.post_attention_layernorm(hidden_states)')
 sys.path.insert(0,str(source));module=types.ModuleType('_duquant_oft_decoder_compat_v1');module.__file__=str(file)+'[independent API/dtype shim]';exec(compile(text,module.__file__,'exec'),module.__dict__)
 original=module.QuantLlamaDecoderLayer
 class Decoder(original):
  def forward(self,*args,**kwargs):
   if len(args)>3 or kwargs.get('past_key_value') is not None or kwargs.get('use_cache',False):raise ValueError('Cache mode not admitted by this compatibility shim')
   return super().forward(*args,**kwargs)
 return Decoder
