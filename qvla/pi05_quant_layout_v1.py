"""Explicit shape policy for the existing 414-target pi0.5 PyTorch cohort.

AWQ, OmniQuant LWC and DuQuant's block transformations require divisible
groups. Preserve group128 except the 27 SigLIP fc2 inputs of width4304,
where the greatest dividing power-of-two group is16. This metadata contract
does not apply quantization or admit any full model. GPTQ keeps its already
registered ragged group128 recipe; existing SmoothQuant stays unchanged.
"""
from math import gcd

def divisible_group(in_features,requested=128):
    if type(in_features) is not int or in_features<=0:raise ValueError('Positive actual input width required')
    if type(requested) is not int or requested<=0 or requested&(requested-1):raise ValueError('Power-of-two requested group required')
    return gcd(in_features,requested)

def layout(targets):
    if len(targets)!=414 or len({x['name'] for x in targets})!=414:raise ValueError('Wrong cohort target scope')
    result=[]
    for row in targets:
        shape=row['shape']
        if len(shape)!=2 or any(type(x) is not int or x<=0 for x in shape):raise ValueError('Invalid Linear shape')
        width=shape[1];group=divisible_group(width)
        if group!=128 and not (row['name'].endswith('.mlp.fc2') and '.vision_tower.' in row['name'] and width==4304 and group==16):
            raise ValueError('Unexpected nondivisible target; do not silently change a new model recipe')
        result.append(dict(name=row['name'],shape=shape,awq_group_size=group,omni_lwc_group_size=group,duquant_rotation_block=group,gptq_group_size=128,gptq_ragged_last_group=width%128!=0))
    if sum(x['awq_group_size']==16 for x in result)!=27:raise ValueError('Unexpected SigLIP fc2 count')
    return result
