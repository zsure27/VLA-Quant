"""Optional BF16 bridge diagnostic, never a quantizer or an admitted backend.

Match the pinned JAX Gemma attention's query scaling and FP32-result QK
contraction. Invoke only in a separately registered, single-model GPU probe.
The original bridge's RMSE/max-absolute thresholds remain unchanged.
"""
import inspect


def install():
    import torch
    from transformers.models.gemma import modeling_gemma
    original = modeling_gemma.eager_attention_forward
    source = inspect.getsource(original)
    if 'torch.matmul(query, key_states.transpose(2, 3)) * scaling' not in source:
        raise RuntimeError('Unexpected eager-attention implementation; inspect before applying probe')
    counts = {'calls': 0}

    def aligned(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if module.training or dropout != 0.0:
            raise RuntimeError('This alignment probe is inference-only')
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise RuntimeError('Unexpected attention tensor precision')
        key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
        # Native JAX: BF16 q *= head_dim**-0.5, then preferred_element_type=FP32.
        scaled_query = query * scaling
        logits = torch.matmul(scaled_query.float(), key_states.float().transpose(2, 3))
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, :key_states.shape[-2]].float()
        weights = torch.softmax(logits, dim=-1).to(query.dtype)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        counts['calls'] += 1
        return output, weights

    modeling_gemma.eager_attention_forward = aligned
    return counts, original
