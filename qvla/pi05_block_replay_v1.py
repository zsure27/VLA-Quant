"""Independent native-block calibration replay utilities; no quantization algorithm.

Stores only the current layer's KV pair. Every replay gets its own cache instance,
so AWQ candidate evaluation cannot accidentally append tokens across candidates.
Full-model admission is a separate required step.
"""
import torch


def _copy_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, tuple):
        return tuple(_copy_tree(x, device) for x in value)
    if isinstance(value, list):
        return [_copy_tree(x, device) for x in value]
    if isinstance(value, dict):
        return {k: _copy_tree(v, device) for k, v in value.items()}
    raise TypeError('Unsupported native block input: '+type(value).__name__)


class LayerKVCache:
    """The getitem/update interface used by the pinned native Gemma attention."""
    def __init__(self, layer_index, pair=None):
        self.layer_index = layer_index
        self.pair = pair

    def __getitem__(self, index):
        if index != self.layer_index or self.pair is None:
            raise KeyError('Only the captured layer may read this cache')
        return self.pair

    def update(self, key, value, layer_index, cache_kwargs=None):
        if layer_index != self.layer_index:
            raise KeyError('Wrong layer in native block replay')
        if self.pair is not None:
            key = torch.cat([self.pair[0], key], dim=2)
            value = torch.cat([self.pair[1], value], dim=2)
        self.pair = (key, value)
        return self.pair


def capture_call(args, kwargs, layer_index):
    """Capture before forward, including cache before the native update."""
    kwargs = dict(kwargs)
    if len(args) > 1 or (args and 'hidden_states' in kwargs):
        raise ValueError('Ambiguous or unsupported positional block arguments')
    hidden = args[0] if args else kwargs.pop('hidden_states')
    if not isinstance(hidden, torch.Tensor) or not torch.isfinite(hidden).all():
        raise ValueError('Malformed calibration hidden state')
    cache = kwargs.pop('past_key_value', None)
    pair = None
    if cache is not None:
        try:
            pair = cache[layer_index]
        except (KeyError, IndexError):
            if not kwargs.get('use_cache', False):
                raise ValueError('Read-only denoising cache is missing its prefix')
    return dict(schema_version=1, layer_index=layer_index,
                hidden=_copy_tree(hidden, 'cpu'), kwargs=_copy_tree(kwargs, 'cpu'),
                cache_present=cache is not None, cache_pair=_copy_tree(pair, 'cpu'))


@torch.no_grad()
def replay_call(module, captured):
    if module.training or captured.get('schema_version') != 1:
        raise ValueError('Replay requires eval mode and a recognized capture')
    device = next(module.parameters()).device
    kwargs = _copy_tree(captured['kwargs'], device)
    pair = _copy_tree(captured['cache_pair'], device)
    kwargs['past_key_value'] = LayerKVCache(captured['layer_index'], pair) if captured['cache_present'] else None
    return module(_copy_tree(captured['hidden'], device), **kwargs)
