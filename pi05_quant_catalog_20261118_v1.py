"""Command identity only. Actual native-model admission is independently required."""
from qvla.infra.registry import QUANTIZERS, register_quantizer
if 'omniquant' not in QUANTIZERS:
    register_quantizer('omniquant', [4], [16])
elif QUANTIZERS['omniquant']['weight_bits'] != [4] or QUANTIZERS['omniquant']['activation_bits'] != [16]:
    raise ValueError('Conflicting OmniQuant registration')
