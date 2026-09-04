from __future__ import annotations

import hashlib
import inspect
import math
import sys
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


PROFILE_FORMAT = 2
AWQ_GROUP_SIZE = 128


def _source_parent(root: Path, package: str, repository: str) -> Path:
    candidates = (root / repository, root)
    for candidate in candidates:
        if (candidate / package).is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find package {package!r} under {root} or {root / repository}"
    )


def load_official_functions(official_root: Path) -> dict[str, Any]:
    awq_parent = _source_parent(official_root, "awq", "llm-awq")
    smooth_parent = _source_parent(official_root, "smoothquant", "smoothquant")
    for path in (smooth_parent, awq_parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    # Python 伪量化器不依赖可选的 AWQ 编译内核。
    sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))

    from awq.quantize.auto_clip import auto_clip_layer
    from awq.quantize.auto_scale import get_act_scale
    from awq.quantize.quantizer import pseudo_quantize_tensor
    from smoothquant.fake_quant import (
        quantize_activation_per_token_absmax,
        quantize_weight_per_channel_absmax,
    )
    from smoothquant.smooth import smooth_ln_fcs, smooth_ln_fcs_llama_like

    functions = {
        "awq_parent": awq_parent,
        "smoothquant_parent": smooth_parent,
        "awq_quantize": pseudo_quantize_tensor,
        "awq_act_scale": get_act_scale,
        "awq_auto_clip": auto_clip_layer,
        "smooth_weight_quantize": quantize_weight_per_channel_absmax,
        "smooth_activation_quantize": quantize_activation_per_token_absmax,
        "smooth_ln_fcs": smooth_ln_fcs,
        "smooth_ln_fcs_llama_like": smooth_ln_fcs_llama_like,
    }
    # 防止 sys.modules 缓存让程序静默导入其他环境中的同名包。
    for key, function in functions.items():
        if not callable(function):
            continue
        source = Path(inspect.getsourcefile(inspect.unwrap(function))).resolve()
        expected = awq_parent if key.startswith("awq_") else smooth_parent
        if source != expected and expected not in source.parents:
            raise RuntimeError(
                f"Imported {key} from unexpected source: {source}; expected under {expected}"
            )
    return functions


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_targets(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"No targets found in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate targets found in {path}")
    return names


def scope_counts(names: list[str]) -> dict[str, int]:
    primary = "vision_backbone.featurizer."
    fused = "vision_backbone.fused_featurizer."
    language = "language_model."
    result = {
        "total": len(names),
        "primary_vision": sum(name.startswith(primary) for name in names),
        "fused_vision": sum(name.startswith(fused) for name in names),
        "language": sum(name.startswith(language) for name in names),
    }
    result["vision"] = result["primary_vision"] + result["fused_vision"]
    result["unexpected"] = result["total"] - result["vision"] - result["language"]
    return result


def validate_targets(model: torch.nn.Module, names: list[str]) -> dict[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    selected: dict[str, torch.nn.Module] = {}
    for name in names:
        if name not in modules:
            raise KeyError(f"Target not found in model: {name}")
        module = modules[name]
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError(f"Unsupported target {name}: {type(module)}")
        selected[name] = module
    return selected


def flatten_weight(module: torch.nn.Module) -> torch.Tensor:
    return module.weight.detach().float().reshape(module.weight.shape[0], -1)


def module_rows(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return value.detach().reshape(-1, value.shape[-1])
    if isinstance(module, torch.nn.Conv2d):
        unfolded = F.unfold(
            value.detach().float(),
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
    raise TypeError(type(module))


def input_channels(module: torch.nn.Module) -> int:
    if isinstance(module, torch.nn.Linear):
        return module.in_features
    if isinstance(module, torch.nn.Conv2d):
        return module.in_channels
    raise TypeError(type(module))


def input_absmax(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return value.detach().float().reshape(-1, value.shape[-1]).abs().amax(dim=0)
    if isinstance(module, torch.nn.Conv2d):
        return value.detach().float().abs().amax(dim=(0, 2, 3))
    raise TypeError(type(module))


def pad_columns(value: torch.Tensor, multiple: int = AWQ_GROUP_SIZE) -> tuple[torch.Tensor, int]:
    columns = value.shape[-1]
    padded = math.ceil(columns / multiple) * multiple
    if padded == columns:
        return value, columns
    return F.pad(value, (0, padded - columns)), columns


def awq_fake_quantize(
    weight: torch.Tensor,
    official: dict[str, Any],
    bits: int = 4,
    group_size: int = AWQ_GROUP_SIZE,
) -> torch.Tensor:
    padded, columns = pad_columns(weight, group_size)
    quantized = official["awq_quantize"](
        padded,
        n_bit=bits,
        zero_point=True,
        q_group_size=group_size,
    )
    return quantized[..., :columns]


@torch.no_grad()
def search_awq_scale(
    weight: torch.Tensor,
    rows: torch.Tensor,
    official: dict[str, Any],
    bits: int,
    group_size: int,
    max_output_rows: int,
) -> tuple[torch.Tensor, float, float]:
    if rows.shape[-1] != weight.shape[-1]:
        raise RuntimeError(f"AWQ input mismatch: {rows.shape} versus {weight.shape}")

    if weight.shape[0] > max_output_rows:
        index = torch.linspace(
            0, weight.shape[0] - 1, max_output_rows, device=weight.device
        ).round().long()
        search_weight = weight[index]
    else:
        search_weight = weight

    reference = F.linear(rows, search_weight)
    denominator = reference.float().square().mean().clamp_min(1e-12)
    activation_scale = official["awq_act_scale"](rows).clamp_min(1e-4)
    best_error = float("inf")
    best_ratio = -1.0
    best_scale: torch.Tensor | None = None

    # 使用官方 AWQ 实现中的 20 点比例搜索网格。
    for index in range(20):
        ratio = index / 20
        scale = activation_scale.pow(ratio).clamp_min(1e-4)
        scale = scale / torch.sqrt(scale.max() * scale.min()).clamp_min(1e-8)
        quantized = awq_fake_quantize(
            search_weight * scale.view(1, -1), official, bits, group_size
        ) / scale.view(1, -1)
        candidate = F.linear(rows, quantized)
        error = float(
            ((candidate - reference).float().square().mean() / denominator).item()
        )
        if error < best_error:
            best_error = error
            best_ratio = ratio
            best_scale = scale.detach().clone()

    if best_scale is None:
        raise RuntimeError("AWQ scale search failed")
    return best_scale, best_ratio, best_error


@torch.no_grad()
def calibrate_awq_entry(
    name: str,
    module: torch.nn.Module,
    rows: torch.Tensor,
    official: dict[str, Any],
    bits: int = 4,
    group_size: int = AWQ_GROUP_SIZE,
    max_output_rows: int = 256,
    clip_tokens: int = 128,
) -> dict[str, Any]:
    weight = flatten_weight(module).to(rows.device)
    rows = rows.float()
    scale, ratio, scale_error = search_awq_scale(
        weight, rows, official, bits, group_size, max_output_rows
    )
    scaled_weight = weight * scale.view(1, -1)
    scaled_rows = rows / scale.view(1, -1)
    padded_weight, original_columns = pad_columns(scaled_weight, group_size)
    padded_rows, _ = pad_columns(scaled_rows, group_size)

    skip_clip = any(token in name for token in ("q_proj", "k_proj", "qkv"))
    clip_max = None
    if not skip_clip:
        original_outputs = padded_weight.shape[0]
        clip_outputs = math.ceil(original_outputs / 64) * 64
        clip_weight = padded_weight
        if clip_outputs != original_outputs:
            clip_weight = F.pad(
                clip_weight, (0, 0, 0, clip_outputs - original_outputs)
            )
        clip_max = official["awq_auto_clip"](
            clip_weight,
            padded_rows,
            n_bit=bits,
            q_config={"zero_point": True, "q_group_size": group_size},
            n_sample_token=min(clip_tokens, padded_rows.shape[0]),
        ).detach()[:original_outputs]
        grouped = padded_weight.reshape(clip_max.shape[0], clip_max.shape[1], -1)
        grouped.clamp_(-clip_max, clip_max)
        padded_weight = grouped.reshape_as(padded_weight)

    quantized = official["awq_quantize"](
        padded_weight,
        n_bit=bits,
        zero_point=True,
        q_group_size=group_size,
    )[..., :original_columns]
    effective = quantized / scale.view(1, -1)
    reference = F.linear(rows, weight)
    candidate = F.linear(rows, effective)
    denominator = reference.float().square().mean().clamp_min(1e-12)
    relative_mse = float(
        ((candidate - reference).float().square().mean() / denominator).item()
    )

    return {
        "module_type": type(module).__name__,
        "shape": list(module.weight.shape),
        "input_scale": scale.cpu().float(),
        "scale_ratio": ratio,
        "scale_search_relative_mse": scale_error,
        "clip_max": None if clip_max is None else clip_max.cpu().float(),
        "clip_skipped_by_official_rule": skip_clip,
        "relative_mse": relative_mse,
    }


@torch.no_grad()
def apply_awq_entry(
    module: torch.nn.Module,
    entry: dict[str, Any],
    official: dict[str, Any],
    bits: int,
    group_size: int,
) -> None:
    weight = flatten_weight(module)
    scale = entry["input_scale"].to(weight.device, torch.float32)
    if scale.numel() != weight.shape[1]:
        raise RuntimeError(f"AWQ scale mismatch: {scale.numel()} != {weight.shape[1]}")
    scaled = weight * scale.view(1, -1)
    padded, columns = pad_columns(scaled, group_size)
    clip_max = entry.get("clip_max")
    if clip_max is not None:
        clip = clip_max.to(padded.device, torch.float32)
        grouped = padded.reshape(clip.shape[0], clip.shape[1], -1)
        grouped.clamp_(-clip, clip)
        padded = grouped.reshape_as(padded)
    quantized = official["awq_quantize"](
        padded,
        n_bit=bits,
        zero_point=True,
        q_group_size=group_size,
    )[..., :columns]
    effective = (quantized / scale.view(1, -1)).reshape_as(module.weight)
    module.weight.copy_(effective.to(module.weight.dtype))


def smooth_groups(names: set[str]) -> list[tuple[str, list[str], str]]:
    groups: list[tuple[str, list[str], str]] = []
    for name in sorted(names):
        if name.endswith(".self_attn.q_proj"):
            prefix = name[: -len(".self_attn.q_proj")]
            targets = [
                prefix + ".self_attn.q_proj",
                prefix + ".self_attn.k_proj",
                prefix + ".self_attn.v_proj",
            ]
            if all(target in names for target in targets):
                groups.append((prefix + ".input_layernorm", targets, name))
        elif name.endswith(".mlp.gate_proj"):
            prefix = name[: -len(".mlp.gate_proj")]
            targets = [prefix + ".mlp.gate_proj", prefix + ".mlp.up_proj"]
            if all(target in names for target in targets):
                groups.append((prefix + ".post_attention_layernorm", targets, name))
        elif name.endswith(".attn.qkv") and ".blocks." in name:
            prefix = name[: -len(".attn.qkv")]
            groups.append((prefix + ".norm1", [name], name))
        elif name.endswith(".mlp.fc1") and ".blocks." in name:
            prefix = name[: -len(".mlp.fc1")]
            groups.append((prefix + ".norm2", [name], name))
    return groups


@torch.no_grad()
def apply_smoothquant_smoothing(
    model: torch.nn.Module,
    entries: dict[str, dict[str, Any]],
    official: dict[str, Any],
    alpha: float,
) -> int:
    modules = dict(model.named_modules())
    count = 0
    for norm_name, target_names, scale_name in smooth_groups(set(entries)):
        if norm_name not in modules:
            raise KeyError(f"SmoothQuant norm not found: {norm_name}")
        norm = modules[norm_name]
        targets = [modules[name] for name in target_names]
        act_scale = entries[scale_name]["activation_absmax"].float()
        if norm.__class__.__name__.endswith("RMSNorm"):
            official["smooth_ln_fcs_llama_like"](norm, targets, act_scale, alpha)
        else:
            official["smooth_ln_fcs"](norm, targets, act_scale, alpha)
        count += 1
    return count


@torch.no_grad()
def smoothquant_weight(module: torch.nn.Module, official: dict[str, Any], bits: int) -> None:
    rows = module.weight.detach().float().reshape(module.weight.shape[0], -1).clone()
    official["smooth_weight_quantize"](rows, n_bits=bits)
    module.weight.copy_(rows.reshape_as(module.weight).to(module.weight.dtype))


def make_smoothquant_activation_hook(
    module: torch.nn.Module,
    quantize: Callable[..., torch.Tensor],
    bits: int,
):
    def hook(_module: torch.nn.Module, args: tuple[Any, ...]):
        if not args:
            raise RuntimeError("Quantized module received no positional input")
        value = args[0]
        if isinstance(module, torch.nn.Linear):
            quantized = quantize(value.detach().clone(), n_bits=bits)
        elif isinstance(module, torch.nn.Conv2d):
            nhwc = value.detach().permute(0, 2, 3, 1).contiguous()
            quantized = quantize(nhwc.clone(), n_bits=bits).permute(0, 3, 1, 2).contiguous()
        else:
            raise TypeError(type(module))
        return (quantized.to(value.dtype),) + tuple(args[1:])

    return hook
