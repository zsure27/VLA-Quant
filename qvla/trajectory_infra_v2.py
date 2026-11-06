"""Explicit CPU interface alias for existing SQ W16 activation-only controls.

Registration is not a numerical gate; all execution uses explicit command binding.
"""
from qvla.infra.registry import register_quantizer
register_quantizer("smoothquant_diagnostic_v2",[4,8,16],[4,8,16])
