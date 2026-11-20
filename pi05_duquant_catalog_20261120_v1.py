"""DuQuant command identity only; not full pi05 or trajectory admission."""
from qvla.infra.registry import QUANTIZERS,register_quantizer
if 'duquant' not in QUANTIZERS:register_quantizer('duquant',[4],[4,8])
assert QUANTIZERS['duquant']['weight_bits']==[4]
assert QUANTIZERS['duquant']['activation_bits']==[4,8]
