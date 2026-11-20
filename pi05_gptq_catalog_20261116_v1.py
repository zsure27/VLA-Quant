"""Command metadata only; real414-layer admission is still required."""
from qvla.infra.registry import QUANTIZERS,register_quantizer
if 'gptq' not in QUANTIZERS:register_quantizer('gptq',[4],[16])
assert QUANTIZERS['gptq']['weight_bits']==[4] and QUANTIZERS['gptq']['activation_bits']==[16]
