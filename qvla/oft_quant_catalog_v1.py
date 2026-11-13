"""Explicit command-backed quantizer names, never a compatibility assertion."""
from qvla.infra.registry import QUANTIZERS, register_quantizer

for name, activations in {
    'gptq': [16], 'omniquant': [16], 'actquant': [16], 'qvla': [16],
    'duquant': [4, 8], 'quantvla': [4, 8], 'holoq_vla': [4],
}.items():
    if name not in QUANTIZERS:
        register_quantizer(name, [4], activations)
    elif QUANTIZERS[name]['weight_bits'] != [4] or QUANTIZERS[name]['activation_bits'] != activations:
        raise ValueError('Existing quantizer registration differs: ' + name)
