"""仅用于 W16A16 定位：整组取消平滑，不能只取消 Q/K/V 的一个分支。"""
import re

CHOICES = ("all", "no-late-attention", "no-late-mlp", "no-late-both", "no-language", "only-primary-vision", "only-fused-vision")


def excluded_norm(name, selection):
    if selection not in CHOICES:
        raise ValueError(selection)
    if selection in ("only-primary-vision", "only-fused-vision"):
        prefix = "vision_backbone.featurizer." if selection == "only-primary-vision" else "vision_backbone.fused_featurizer."
        return not name.startswith(prefix)
    if selection == "no-language":
        return name.startswith("language_model.")
    match = re.fullmatch(
        r"language_model\.model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)", name
    )
    if not match or not 23 <= int(match[1]) <= 31:
        return False
    return selection == "no-late-both" or (
        selection == "no-late-attention" and match[2] == "input_layernorm"
    ) or (selection == "no-late-mlp" and match[2] == "post_attention_layernorm")
