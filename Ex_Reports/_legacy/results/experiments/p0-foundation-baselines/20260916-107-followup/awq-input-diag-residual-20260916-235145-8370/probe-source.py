"""面向现有 OpenVLA-OFT 环境的只读检查点诊断。

每个量化用例都在新进程中运行。这些是伪量化诊断，不是打包的
INT2/INT4 内核，也不能替代官方 AWQ 基线。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_trusted(path):
    # 仅加载自己生成的校准/教师文件，禁止加载来源不明的 .pt 文件。
    return torch.load(path, map_location="cpu", weights_only=False)


def finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite tensor: {name}")


def old_symmetric_row_quantize(weight, bits, clip_ratio=1.0):
    """复刻历史整行对称量化器，仅用于对照，不能称为官方 AWQ。"""
    shape = weight.shape
    rows = weight.reshape(weight.shape[0], -1)
    qmax = (1 << (bits - 1)) - 1
    maximum = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * clip_ratio
    scale = maximum.clamp_min(1e-8) / qmax
    quantized = torch.round(rows / scale).clamp(-(qmax + 1), qmax)
    return (quantized * scale).reshape(shape)


def metric(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    r, q = reference.float(), candidate.float()
    finite(r, "reference")
    finite(q, "candidate")
    mse = (q - r).square().mean().item()
    energy = r.square().mean().item()
    return {
        "mse": mse, "relative_mse": mse / energy if energy > 1e-12 else None,
        "reference_rms": energy ** 0.5, "candidate_rms": q.square().mean().sqrt().item(),
        "cosine": F.cosine_similarity(r.reshape(1, -1), q.reshape(1, -1)).item(),
        "max_abs_error": (q - r).abs().max().item(),
        "reference_zero_fraction": (r == 0).float().mean().item(),
        "candidate_zero_fraction": (q == 0).float().mean().item(),
    }


def official_functions(root):
    from qvla.official_quant_adapter import load_official_functions
    funcs = load_official_functions(root)
    sources = {}
    for key, value in funcs.items():
        if not callable(value):
            continue
        actual = Path(inspect.getsourcefile(inspect.unwrap(value))).resolve()
        parent = funcs["awq_parent" if key.startswith("awq_") else "smoothquant_parent"]
        if parent != actual and parent not in actual.parents:
            raise RuntimeError(f"Imported function is outside requested source tree: {key}: {actual}")
        sources[key] = {"path": str(actual), "sha256": digest(actual)}
    return funcs, sources


def profiles(directory, target_file, checkpoint, sources):
    from qvla.official_quant_adapter import read_targets
    from qvla.run_eval_official_quant import load_profiles
    # 统一正式入口的格式、源码和范围验证，不能让诊断绕过基线合同。
    load_profiles(sorted(directory.glob("*.pt")), "smoothquant")
    entries, first, manifest = {}, None, []
    common = ("format_version", "method", "bits", "checkpoint", "num_samples",
              "sample_names", "group_size", "smooth_alpha", "official_sources",
              "checkpoint_identity", "sample_sha256", "seed", "algorithm", "llm_attention")
    for path in sorted(directory.glob("*.pt")):
        item = load_trusted(path)
        if item.get("format_version") != 3 or item.get("method") != "smoothquant":
            raise ValueError(f"Not a v3 SmoothQuant profile; recalibrate: {path}")
        if first is None:
            first = item
        for key in common:
            if item.get(key) != first.get(key):
                raise ValueError(f"Mixed profile metadata for {key}: {path}")
        overlap = set(entries) & set(item["entries"])
        if overlap:
            raise ValueError(f"Duplicate profile targets: {overlap}")
        for name, entry in item["entries"].items():
            finite(entry["activation_absmax"], name)
            if (entry["activation_absmax"] < 0).any():
                raise ValueError(f"Negative activation absmax: {name}")
        entries.update(item["entries"])
        manifest.append({"file": str(path), "sha256": digest(path)})
    if first is None:
        raise ValueError(f"No profiles in {directory}")
    from qvla.baseline_contract import checkpoint_identity
    if first["checkpoint_identity"] != checkpoint_identity(checkpoint):
        raise ValueError("Profile checkpoint contents differ")
    if set(entries) != set(read_targets(target_file)):
        raise ValueError("Profile target names differ from the complete target file")
    for saved, loaded in (("smoothquant_smooth", "smooth_ln_fcs"),
                          ("smoothquant_fake_quant", "smooth_activation_quantize")):
        if first["official_sources"][saved]["sha256"] != sources[loaded]["sha256"]:
            raise ValueError(f"Source changed since calibration: {saved}")
    return entries, first, manifest


def initialize_readonly(checkpoint, seed):
    # 上游加载器通常会重写本地检查点中的 config.json/模型源码；诊断过程禁止这样做。
    # 本地模型类的注册仍由原始初始化器完成。
    import experiments.robot.openvla_utils as utils
    from qvla.action_jacobian_batch import initialize
    originals = {}
    for name in ("update_auto_map", "check_model_logic_mismatch"):
        if hasattr(utils, name):
            originals[name] = getattr(utils, name)
            setattr(utils, name, lambda *a, **k: None)
    try:
        return initialize(checkpoint, seed)
    finally:
        for name, fn in originals.items():
            setattr(utils, name, fn)


def select_scope(name, scope, smoothed):
    return {
        "all": True, "none": False,
        "vision": name.startswith("vision_backbone."),
        "language": name.startswith("language_model."),
        "primary": name.startswith("vision_backbone.featurizer."),
        "fused": name.startswith("vision_backbone.fused_featurizer."),
        "attention": ".attn." in name or ".self_attn." in name,
        "mlp": ".mlp." in name,
        "patch": ".patch_embed." in name,
        "no-patch": ".patch_embed." not in name,
        "smoothed": name in smoothed,
        "unsmoothed": name not in smoothed,
    }[scope]


def evenly(length, maximum, device="cpu"):
    return torch.linspace(0, length - 1, min(length, maximum), device=device).round().long().unique()


class Recorder:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.reset(None)
        original = model._regression_or_discrete_prediction
        self.original_prediction = original

        def prediction(*args, **kwargs):
            bound = inspect.signature(original).bind(*args, **kwargs).arguments
            self.action_start = int(bound["NUM_PATCHES"] + bound["NUM_PROMPT_TOKENS"])
            from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
            self.action_count = ACTION_DIM * NUM_ACTIONS_CHUNK
            return original(*args, **kwargs)
        model._regression_or_discrete_prediction = prediction

        def rescue_hook(_module, _inputs, output):
            if self.rescue is None:
                return None
            teacher = self.rescue.to(output.device, output.dtype)
            if teacher.shape != output.shape:
                raise ValueError("Oracle projector shape mismatch")
            return teacher
        self.handles.append(model.projector.register_forward_hook(rescue_hook))
        self.handles.append(model.projector.register_forward_hook(self.hook("projector")))
        self.handles.append(model.vision_backbone.register_forward_hook(self.hook("vision_output")))
        for name, module in model.named_modules():
            if re.fullmatch(r"language_model\.model\.layers\.\d+", name):
                self.handles.append(module.register_forward_hook(self.hook(name)))
            elif re.fullmatch(r"vision_backbone\.(?:fused_)?featurizer\.blocks\.\d+", name):
                self.handles.append(module.register_forward_hook(self.hook(name)))
        first = model.language_model.model.layers[0]

        def llm_input(_module, args, kwargs):
            value = args[0] if args else kwargs["hidden_states"]
            self.capture("llm_input", value)
        self.handles.append(first.register_forward_pre_hook(llm_input, with_kwargs=True))

    def reset(self, rescue):
        self.traces, self.counts, self.projector = {}, {}, None
        self.rescue = rescue
        self.action_start, self.action_count = None, None

    def hook(self, name):
        def hook(_module, _inputs, output):
            self.capture(name, output[0] if isinstance(output, tuple) else output)
        return hook

    def capture(self, name, tensor):
        value = tensor.detach()
        finite(value, name)
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"Expected batch=1 sequence features at {name}, got {value.shape}")
        call = self.counts.get(name, 0)
        self.counts[name] = call + 1
        if name == "projector":
            self.projector = value.cpu().clone()
        length = value.shape[1]
        if name.startswith("language_model.") or name == "llm_input":
            patches = self.model.vision_backbone.get_num_patches()
            if self.model.vision_backbone.get_num_images_in_input() != 2:
                raise ValueError("This diagnostic expects two cameras")
            if self.action_start is None:
                raise RuntimeError("Action readout slice was not captured")
            ranges = {
                "image_main": (1, 1 + patches),
                "image_wrist": (1 + patches, 1 + 2 * patches),
                "proprio": (1 + 2 * patches, 2 + 2 * patches),
                "text": (2 + 2 * patches, self.action_start),
                "action_readout": (self.action_start, self.action_start + self.action_count),
            }
        else:
            ranges = {"features": (0, length)}
        for group, (lo, hi) in ranges.items():
            if not 0 <= lo <= hi <= length:
                raise ValueError(f"Invalid token slice: {name}, {group}: {lo}:{hi}/{length}")
            if hi == lo:
                continue
            # 保留完整通道向量，仅抽样 token；动作读出 token 全部保留。
            maximum = self.action_count if group == "action_readout" else 16
            ids = evenly(hi - lo, maximum, value.device) + lo
            key = f"{name}@{call}/{group}"
            self.traces[key] = value[:, ids, :].cpu().clone()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model._regression_or_discrete_prediction = self.original_prediction


def predict(sample_path, cfg, model, action_head, proprio_projector, processor):
    from qvla.action_jacobian_batch import load_sample, prepare_inputs
    sample = load_sample(sample_path)
    inputs, state = prepare_inputs(sample, cfg, model, processor, torch.device("cuda:0"))
    input_hashes = {name: hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for name, value in inputs.items()}
    input_hashes["normalized_proprio"] = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
    raw, hidden = model.predict_action(
        **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=state,
        proprio_projector=proprio_projector, action_head=action_head,
        noisy_action_projector=None, use_film=False,
    )
    normalized = action_head.predict_action(hidden).reshape(-1, 7).detach().float().cpu()
    return torch.as_tensor(raw).float(), normalized, input_hashes


@torch.no_grad()
def awq_micro(args, bundle, official):
    from qvla.official_quant_adapter import module_rows, flatten_weight, awq_fake_quantize
    from qvla.official_quant_adapter import calibrate_awq_entry, apply_awq_entry
    cfg, model, head, proprio, processor = bundle
    modules = dict(model.named_modules())
    names = [n.strip() for n in args.micro_targets.split(",") if n.strip()]
    captured = {n: [] for n in names}
    handles = []
    sample_index = 0
    for name in names:
        module = modules[name]
        def hook(m, inputs, _out, name=name):
            rows = module_rows(m, inputs[0])
            chosen = rows[evenly(len(rows), 128, rows.device)].float().cpu()
            captured[name].append((sample_index, chosen))
        handles.append(module.register_forward_hook(hook))
    try:
        for sample_index, path in enumerate(args.samples):
            predict(path, cfg, model, head, proprio, processor)
    finally:
        for handle in handles:
            handle.remove()
    split = len(args.samples) // 2
    if split < 1 or len(args.samples) - split < 1:
        raise ValueError("AWQ microprobe needs at least two distinct samples")
    results = {}
    for name in names:
        module = modules[name]
        calibration = torch.cat([x for i, x in captured[name] if i < split]).cuda()
        heldout = torch.cat([x for i, x in captured[name] if i >= split]).cuda()
        original = module.weight.detach().cpu().clone()
        weight = flatten_weight(module).clone()
        reference = F.linear(heldout, weight)
        values = {}
        try:
            old = old_symmetric_row_quantize(weight, 2, 1.0)
            values["old_symmetric_row_rtn_w2"] = metric(reference, F.linear(heldout, old))
            del old
            for bits in (2, 4):
                rtn = awq_fake_quantize(weight, official, bits, 128)
                values[f"official_asymmetric_group128_rtn_w{bits}"] = metric(reference, F.linear(heldout, rtn))
                del rtn
                entry = calibrate_awq_entry(name, module, calibration, official, bits=bits)
                # 使用已有的序列化路径，直接测量实际应用后的结果。
                apply_awq_entry(module, entry, official, bits, 128)
                values[f"adapter_linear_search_w{bits}_NOT_official_block_awq"] = metric(
                    reference, F.linear(heldout, flatten_weight(module)))
                values[f"adapter_linear_search_w{bits}_NOT_official_block_awq"]["calibration_relative_mse"] = entry["relative_mse"]
                module.weight.copy_(original.to(module.weight.device))
                del entry
            results[name] = values
            print(f"[microprobe] {name}", flush=True)
        finally:
            module.weight.copy_(original.to(module.weight.device))
        del weight, reference, calibration, heldout
        torch.cuda.empty_cache()
    save_json(args.output / "awq_micro.json", {
        "note": "Local Linear-output MSE, not action error or an official full-block AWQ benchmark",
        "calibration_samples": [str(p) for p in args.samples[:split]],
        "heldout_samples": [str(p) for p in args.samples[split:]], "results": results,
    })


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples-dir", required=True, type=Path)
    p.add_argument("--official-root", required=True, type=Path)
    p.add_argument("--targets-file", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=("teacher", "repeat", "smoothquant", "awq", "awq-micro"), required=True)
    p.add_argument("--awq-profile", type=Path)
    p.add_argument("--awq-disable-clip", choices=("none", "all", "attention", "mlp"), default="none")
    p.add_argument("--awq-w4-layers", default="")
    p.add_argument("--awq-w4-profile", type=Path)
    p.add_argument("--awq-rescue-family", choices=("attention", "mlp", "all"))
    p.add_argument("--awq-rescue-layers", default="")
    p.add_argument("--awq-residual-rank", type=int, default=0)
    p.add_argument("--awq-residual-layers", default="")
    p.add_argument("--awq-residual-calibration-dir", type=Path)
    p.add_argument("--awq-residual-token-scope", choices=("all", "action"), default="all")
    p.add_argument("--smoothing-pairs-fp32", action="store_true")
    p.add_argument("--smoothing-bypass", action="store_true")
    p.add_argument("--export-candidate-teacher", action="store_true")
    p.add_argument("--teacher-fp32-reference", action="store_true")
    p.add_argument("--awq-vision-bits", type=int, choices=(2, 4, 16), default=16)
    p.add_argument("--awq-vision-profile", type=Path)
    p.add_argument("--awq-vision-branch", choices=("all", "primary", "fused"), default="all")
    p.add_argument("--awq-primary-no-clip", action="store_true")
    p.add_argument("--awq-primary-group64-profile", type=Path)
    p.add_argument("--attention-layers", default="", help="例如 7,15,23,31；不切换 attention 后端")
    p.add_argument("--teacher-dir", type=Path)
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--offset", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--weight-bits", type=int, choices=(2, 4, 8, 16), default=4)
    p.add_argument("--activation-bits", type=int, choices=(4, 8, 16), default=4)
    scopes = ("all", "none", "vision", "language", "primary", "fused", "attention", "mlp",
              "patch", "no-patch", "smoothed", "unsmoothed")
    p.add_argument("--weight-scope", choices=scopes, default="all")
    p.add_argument("--activation-scope", choices=scopes, default="all")
    p.add_argument("--activation-math-fp32", action="store_true")
    p.add_argument("--oracle-projector", action="store_true")
    from smoothing_selection import CHOICES, excluded_norm
    p.add_argument("--smoothing-selection", choices=CHOICES, default="all",
                   help="W16A16 因果消融；整组关闭第 23–31 层的平滑，不改变 attention 本身")
    p.add_argument("--alpha", type=float)
    p.add_argument("--micro-targets", default=",".join((
        "vision_backbone.featurizer.patch_embed.proj",
        "vision_backbone.featurizer.blocks.11.mlp.fc2",
        "vision_backbone.fused_featurizer.blocks.12.attn.qkv",
        "vision_backbone.fused_featurizer.blocks.12.mlp.fc2",
        "language_model.model.layers.15.self_attn.q_proj",
        "language_model.model.layers.15.self_attn.o_proj",
        "language_model.model.layers.15.mlp.gate_proj",
        "language_model.model.layers.15.mlp.down_proj")))
    args = p.parse_args()
    if args.smoothing_pairs_fp32 and (args.mode != "smoothquant" or args.weight_bits != 16 or args.activation_bits != 16 or
        args.smoothing_selection not in ("no-language", "only-primary-vision", "only-fused-vision")):
        p.error("FP32 pairs are vision-only smoothing W16A16 numerical diagnostics")
    if (args.smoothing_bypass or args.export_candidate_teacher or args.teacher_fp32_reference) and not args.smoothing_pairs_fp32:
        p.error("Numerical reference flags require FP32 vision pairs")
    from awq_interventions import parse_layers, plan, vision_plan, in_vision_branch, remove_primary_clips, primary_group_plan, family_precision_plan
    w4_layers = parse_layers(args.awq_w4_layers)
    rescue_layers = parse_layers(args.awq_rescue_layers)
    residual_layers = parse_layers(args.awq_residual_layers)
    if bool(residual_layers) != bool(args.awq_residual_rank):
        p.error("Residual recovery requires rank and layers together")
    if (args.awq_residual_calibration_dir is not None or args.awq_residual_token_scope != "all") and not residual_layers:
        p.error("Residual input statistics require a residual branch")
    if args.awq_residual_token_scope == "action" and args.awq_residual_calibration_dir is None:
        p.error("Action-token residual statistics require calibration inputs")
    if residual_layers and (args.awq_residual_rank not in (4,8,16) or args.mode != "awq" or
            args.weight_bits != 2 or args.activation_bits != 16 or args.weight_scope != "language" or
            args.awq_disable_clip != "all" or rescue_layers or w4_layers or args.awq_w4_profile or args.awq_vision_bits != 16):
        p.error("Residual probe requires language W2 no-clip, BF16 vision, no mixed precision")
    if bool(rescue_layers) != bool(args.awq_rescue_family):
        p.error("Family rescue requires both layer selection and family")
    if rescue_layers and (w4_layers or args.awq_w4_profile or args.awq_vision_bits != 16 or args.awq_disable_clip != "all"):
        p.error("Family precision rescue retains W2 scales, no clipping and BF16 vision; no W4 peer")
    vision_composition = args.awq_vision_bits != 16
    if args.awq_primary_group64_profile and (args.awq_vision_bits != 2 or args.awq_vision_branch not in ('primary', 'all') or args.awq_primary_no_clip):
        p.error('G64对照仅允许primary/all分支W2保留裁剪')
    if args.awq_primary_no_clip and (args.awq_vision_bits != 2 or args.awq_vision_branch != "primary"):
        p.error("主视觉无裁剪仅允许primary分支W2")
    if args.awq_vision_branch != "all" and not vision_composition:
        p.error("分支选择仅用于显式视觉量化组合")
    if (args.awq_vision_bits == 4) != (args.awq_vision_profile is not None):
        p.error("视觉 W4 必须提供独立 profile，其他视觉位宽不提供该参数")
    if vision_composition and (args.awq_disable_clip != "all" or w4_layers):
        p.error("视觉组合固定语言 W2 全部取消裁剪，不叠加语言 W4 层保护")
    intervention = args.awq_disable_clip != "none" or bool(w4_layers) or args.awq_w4_profile is not None or vision_composition or bool(rescue_layers)
    if intervention and (args.mode != "awq" or args.weight_scope != ("all" if vision_composition else "language") or
                         args.weight_bits != 2 or args.activation_bits != 16 or args.oracle_projector):
        p.error("AWQ 干预要求语言 W2A16，无 oracle；视觉组合范围用 all，纯语言用 language")
    if bool(w4_layers) != (args.awq_w4_profile is not None):
        p.error("W4 层和独立 profile 必须配对")
    if args.smoothing_selection != "all" and (
        args.mode != "smoothquant" or args.weight_bits != 16 or args.activation_bits != 16
    ):
        p.error("平滑范围消融仅允许 smoothquant W16A16")
    attention_layers = sorted(set(int(x) for x in args.attention_layers.split(",") if x))
    if any(i < 0 or i >= 32 for i in attention_layers):
        raise ValueError("attention 层号必须为 0..31")
    if args.alpha is not None and not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if args.output.exists():
        raise ValueError(f"Output already exists; use a new directory: {args.output}")
    args.output.mkdir(parents=True)
    if args.offset < 0 or args.num_samples < 1:
        raise ValueError("Invalid sample range")
    args.samples = sorted(args.samples_dir.glob("sample-*.npz"))[args.offset:args.offset + args.num_samples]
    if len(args.samples) != args.num_samples:
        raise ValueError("Insufficient samples; adjust --offset/--num-samples explicitly")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    official, sources = official_functions(args.official_root)
    entries, meta, profile_manifest = {}, {}, []
    if args.mode == "smoothquant":
        if args.weight_bits == 2:
            raise ValueError("SQ W2 不在本轮基线范围")
        if args.profile_dir is None or args.teacher_dir is None:
            raise ValueError("SmoothQuant requires --profile-dir and --teacher-dir")
        entries, meta, profile_manifest = profiles(args.profile_dir, args.targets_file, args.checkpoint, sources)
        overlap = set(meta["sample_names"]) & {s.name for s in args.samples}
        if overlap:
            raise ValueError(f"Probe/calibration sample overlap: {overlap}")
    if args.mode == "awq":
        if args.awq_profile is None or args.teacher_dir is None:
            raise ValueError("AWQ 需要 --awq-profile 与 --teacher-dir")
        from qvla.run_eval_official_quant import load_profiles
        from qvla.baseline_contract import checkpoint_identity
        entries, meta = load_profiles([args.awq_profile], "awq")
        if meta["checkpoint_identity"] != checkpoint_identity(args.checkpoint):
            raise ValueError("AWQ checkpoint 指纹不符")
        if set(meta["sample_names"]) & {s.name for s in args.samples}:
            raise ValueError("AWQ 校准与探针帧重叠")
        if args.weight_scope not in ("all", "vision", "language") or args.activation_scope != "all":
            raise ValueError("AWQ 首轮只支持 all/vision/language；细粒度恢复另做受控实验")
        profile_manifest = [{"file": str(args.awq_profile), "sha256": digest(args.awq_profile)}]
        if intervention:
            w4 = load_profiles([args.awq_w4_profile], "awq") if w4_layers else None
            intervention_plan = (family_precision_plan(entries, meta, rescue_layers, args.awq_rescue_family)
                                 if rescue_layers else plan(entries, meta, args.awq_disable_clip, w4_layers, w4))
            if w4:
                profile_manifest.append({"file": str(args.awq_w4_profile), "sha256": digest(args.awq_w4_profile)})
            if vision_composition:
                peer = load_profiles([args.awq_vision_profile], "awq") if args.awq_vision_bits == 4 else None
                visual_targets = vision_plan(entries, meta, args.awq_vision_bits, peer, args.awq_vision_branch)
                if args.awq_primary_group64_profile:
                    visual_targets.update(primary_group_plan(entries, meta, load_profiles([args.awq_primary_group64_profile], 'awq')))
                    profile_manifest.append({'file': str(args.awq_primary_group64_profile), 'sha256': digest(args.awq_primary_group64_profile)})
                visual_removed = []
                if args.awq_primary_no_clip:
                    visual_targets, visual_removed = remove_primary_clips(visual_targets)
                intervention_plan[1].update(visual_targets)
                if peer:
                    profile_manifest.append({"file": str(args.awq_vision_profile), "sha256": digest(args.awq_vision_profile)})
        # profile 文件不变也要核对实际加载的官方源码。
        for saved, loaded in (("awq_auto_scale", "awq_auto_scale_block"),
                              ("awq_auto_clip", "awq_auto_clip"), ("awq_quantizer", "awq_quantize")):
            if meta["official_sources"][saved]["sha256"] != sources[loaded]["sha256"]:
                raise ValueError("AWQ 官方源码与校准时不同")
    bundle = initialize_readonly(args.checkpoint, args.seed)
    cfg, model, head, proprio, processor = bundle
    import transformers
    import timm
    import qvla.official_quant_adapter as adapter_module
    import qvla.action_jacobian_batch as helper_module
    model_source = Path(inspect.getsourcefile(type(model)))
    from qvla.baseline_contract import checkpoint_identity
    full_identity = meta.get("checkpoint_identity") or checkpoint_identity(args.checkpoint)
    if meta and (meta["model_source_sha256"] != digest(model_source) or
                 meta["llm_attention"] != model.language_model.config._attn_implementation):
        raise ValueError("校准/探针模型源码或 attention 后端不同")
    checkpoint_meta = {str(f.relative_to(args.checkpoint)): digest(f) for f in Path(args.checkpoint).glob("*.json")}
    manifest = {
        "fp32_smoothing_pairs": args.smoothing_pairs_fp32,
        "fp32_smoothing_selection": args.smoothing_selection if args.smoothing_pairs_fp32 else None,
        "smoothing_bypassed": args.smoothing_bypass,
        "torch": torch.__version__, "transformers": transformers.__version__, "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(),
        "adapter_sha256": digest(adapter_module.__file__), "helper_sha256": digest(helper_module.__file__),
        "model_class": str(type(model)), "model_source_sha256": digest(model_source),
        "llm_attention": getattr(model.language_model.config, "_attn_implementation", None),
        "vision_fused_attention": {n: bool(m.fused_attn) for n, m in model.named_modules() if hasattr(m, "fused_attn")},
        "checkpoint_json_sha256": checkpoint_meta, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": full_identity,
        "attention_layers": attention_layers,
        "seed": args.seed,
        "diagnostic_source_sha256": digest(Path(__file__)),
        "awq_intervention_source_sha256": digest(Path(__file__).with_name("awq_interventions.py")),
        "smoothing_selection_source_sha256": digest(Path(__file__).with_name("smoothing_selection.py")),
        "sources": sources, "profiles": profile_manifest,
        "samples": {s.name: digest(s) for s in args.samples},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "samples"},
        "limits": "已校验完整权重内容；离线特征一致性不能替代闭环成功率。",
    }
    save_json(args.output / "manifest.json", manifest)
    if args.mode == "awq-micro":
        awq_micro(args, bundle, official)
        return
    handles = []
    activation_stats, activation_calls = {}, {}
    if args.mode in ("smoothquant", "awq", "repeat"):
        if args.teacher_dir is None:
            raise ValueError("候选用例需要 --teacher-dir")
        teacher_manifest = json.loads((args.teacher_dir / "manifest.json").read_text())
        if bool(teacher_manifest.get("fp32_smoothing_pairs",False)) != args.teacher_fp32_reference:
            raise ValueError("Teacher numerical path must be explicitly selected")
        if args.teacher_fp32_reference and teacher_manifest.get("fp32_smoothing_selection") != args.smoothing_selection:
            raise ValueError("FP32 teacher pair selection mismatch")
        if teacher_manifest["attention_layers"] != attention_layers:
            raise ValueError("教师与候选必须选择相同的 attention 采样层")
        for key in ("torch", "transformers", "timm", "seed", "model_source_sha256", "adapter_sha256", "helper_sha256", "llm_attention", "vision_fused_attention", "checkpoint_json_sha256", "checkpoint_identity", "checkpoint", "samples"):
            if manifest[key] != teacher_manifest[key]:
                raise ValueError(f"Teacher/candidate mismatch: {key}")
    if args.mode == "awq":
        from qvla.awq_block import apply_block_scales, apply_llama_entry
        from qvla.official_quant_adapter import apply_awq_entry
        residual_records = {}
        residual_input_rms = {}
        residual_calibration_manifest = []
        if meta["bits"] != args.weight_bits or args.activation_bits != 16:
            raise ValueError("AWQ 必须 A16，profile bits 须与候选 W 位宽一致")
        modules = dict(model.named_modules())
        w_names = [n for n in entries if select_scope(n, args.weight_scope, set())]
        if vision_composition:
            w_names = [n for n in w_names if n.startswith("language_model.") or in_vision_branch(n, args.awq_vision_branch)]
        if intervention:
            scale_plan, target_plan, removed = intervention_plan
            if set(w_names) != set(target_plan):
                raise ValueError("干预实际目标与声明范围不同")
            for prefix, scales in scale_plan.items():
                apply_block_scales(modules[prefix], scales, official)
            if args.awq_residual_calibration_dir is not None:
                calibration=sorted(args.awq_residual_calibration_dir.glob("sample-*.npz"))[64:72]
                if len(calibration)!=8 or set(p.resolve() for p in calibration)&set(p.resolve() for p in args.samples):
                    raise ValueError("Residual calibration requires eight separate inputs at indices64-71")
                selected_names=[n for n in target_plan if int(n.split(".layers.")[1].split(".")[0]) in residual_layers]
                stats={}
                calibration_recorder=Recorder(model)
                calibration_handles=[]
                def input_stat_hook(name):
                    def collect(_module,inputs):
                        x=inputs[0].detach().float()
                        if args.awq_residual_token_scope=="action":
                            start=calibration_recorder.action_start; count=calibration_recorder.action_count
                            if start is None or count is None or x.ndim!=3 or start+count>x.shape[1]:
                                raise ValueError("Action token indices unavailable for residual statistics")
                            x=x[:,start:start+count,:]
                        x=x.reshape(-1,x.shape[-1])
                        total=x.square().sum(0).cpu()
                        previous=stats.get(name)
                        stats[name]=(total+(previous[0] if previous else 0),x.shape[0]+(previous[1] if previous else 0),1+(previous[2] if previous else 0))
                    return collect
                for n in selected_names: calibration_handles.append(modules[n].register_forward_pre_hook(input_stat_hook(n)))
                np_state=np.random.get_state(); py_state=random.getstate()
                try:
                    with torch.random.fork_rng(devices=[0]):
                        for path in calibration:
                            calibration_recorder.reset(None)
                            predict(path,cfg,model,head,proprio,processor)
                            residual_calibration_manifest.append({"sample":path.name,"sha256":digest(path)})
                            print("RESIDUAL_INPUT_CALIBRATED",path.name,flush=True)
                finally:
                    for h in calibration_handles: h.remove()
                    calibration_recorder.close()
                    np.random.set_state(np_state); random.setstate(py_state)
                if set(stats)!=set(selected_names) or any(v[2]!=8 for v in stats.values()):
                    raise ValueError("Residual input calibration target/call coverage mismatch")
                residual_input_rms={n:(s/c).sqrt() for n,(s,c,_) in stats.items()}
                save_json(args.output/"residual_input_statistics.json",{n:{"rows":stats[n][1],"rms":r.tolist()} for n,r in residual_input_rms.items()})
            for name, (entry, bits, group_size) in target_plan.items():
                if list(modules[name].weight.shape) != entry["shape"]:
                    raise ValueError("干预 profile 参数形状不符")
                fn = apply_llama_entry if name.startswith("language_model.") else apply_awq_entry
                residual_selected = bool(residual_layers) and int(name.split(".layers.")[1].split(".")[0]) in residual_layers
                original_weight = modules[name].weight.detach().clone() if residual_selected else None
                fn(modules[name], entry, official, bits, group_size)
                if residual_selected:
                    from low_rank_recovery import attach_residual
                    details=attach_residual(modules[name], original_weight, args.awq_residual_rank,
                        int(hashlib.sha256(name.encode()).hexdigest()[:8],16),residual_input_rms.get(name))
                    residual_records[name]=details
                    print("RESIDUAL_ATTACHED",name,flush=True)
                    del original_weight
        elif args.weight_scope != "vision":
            for prefix, scales in meta["block_scales"].items():
                apply_block_scales(modules[prefix], scales, official)
        for name in ([] if intervention else w_names):
            fn = apply_llama_entry if name.startswith("language_model.") else apply_awq_entry
            fn(modules[name], entries[name], official, args.weight_bits, meta["group_size"])
        save_json(args.output / "scope.json", {"weight_targets": w_names, "activation_targets": [],
            "low_rank_residual": residual_records,
            "residual_adapter_parameters": sum(r["adapter_parameters"] for r in residual_records.values()),
            "residual_training_steps": 0,
            "residual_input_token_scope":args.awq_residual_token_scope,
            "residual_calibration_manifest":residual_calibration_manifest,
            "intervention": intervention, "disable_clip": args.awq_disable_clip,
            "rescue_family": args.awq_rescue_family, "rescue_layers": sorted(rescue_layers),
            "rescue_coordinates": "original_W2_no_clip" if rescue_layers else None,
            "removed_clip_targets": removed if intervention else [],
            "w4_layers": sorted(w4_layers),
            "vision_composition": vision_composition,
            "vision_weight_bits": args.awq_vision_bits if vision_composition else None,
            "vision_branch": args.awq_vision_branch if vision_composition else None,
            "primary_no_clip": args.awq_primary_no_clip,
            "group_size_by_target": {n: target_plan[n][2] if intervention else meta['group_size'] for n in w_names},
            "removed_visual_clip_targets": visual_removed if vision_composition else [],
            "clip_intervention_scope": ("language_and_primary" if args.awq_primary_no_clip else "language_only") if intervention else None,
            "weight_bits_by_target": {n: target_plan[n][1] if intervention else args.weight_bits for n in w_names},
            "recipe": meta["algorithm"], "warning": "vision 为显式适配；范围消融不是完整论文基线"})
    if args.mode == "smoothquant":
        from qvla.official_quant_adapter import apply_smoothquant_smoothing, smooth_groups
        from qvla.official_quant_adapter import smoothquant_weight, make_smoothquant_activation_hook, module_rows
        modules = dict(model.named_modules())
        for name, entry in entries.items():
            if list(modules[name].weight.shape) != entry["shape"]:
                raise ValueError(f"Profile shape mismatch: {name}")
        alpha = meta["smooth_alpha"] if args.alpha is None else args.alpha
        groups = smooth_groups(set(entries))
        skipped = [g for g in groups if excluded_norm(g[0], args.smoothing_selection)]
        omitted = {n for _, names, _ in skipped for n in names}
        smoothing_entries = {n: e for n, e in entries.items() if n not in omitted}
        fp32_pairs=[]
        if args.smoothing_pairs_fp32:
            from fp32_smoothing_pairs import promote_pairs
            pair_handles,fp32_pairs=promote_pairs(model,smooth_groups(set(smoothing_entries)))
            handles.extend(pair_handles)
        count = 0 if args.smoothing_bypass else apply_smoothquant_smoothing(model, smoothing_entries, official, alpha)
        smoothed = set() if args.smoothing_bypass else {n for _, names, _ in smooth_groups(set(smoothing_entries)) for n in names}
        w_names, a_names = [], []
        quantize = official["smooth_activation_quantize"]
        if args.activation_math_fp32:
            base = quantize
            def quantize(value, n_bits):
                return base(value.float(), n_bits=n_bits).to(value.dtype)
        for name in entries:
            module = modules[name]
            if args.weight_bits < 16 and select_scope(name, args.weight_scope, smoothed):
                smoothquant_weight(module, official, args.weight_bits)
                w_names.append(name)
            if args.activation_bits < 16 and select_scope(name, args.activation_scope, smoothed):
                inner = make_smoothquant_activation_hook(module, quantize, args.activation_bits)
                def observed_hook(m, values, inner=inner, name=name):
                    result = inner(m, values)
                    before, after = module_rows(m, values[0]), module_rows(m, result[0])
                    ids = evenly(len(before), 16, before.device)
                    before, after = before[ids].float(), after[ids].float()
                    call = activation_calls.get(name, 0)
                    activation_calls[name] = call + 1
                    stats = metric(before, after)
                    ratios = before.abs().amax(-1) / before.square().mean(-1).sqrt().clamp_min(1e-12)
                    stats["max_to_rms_mean"] = ratios.mean().item()
                    stats["nonzero_to_zero_fraction"] = ((before != 0) & (after == 0)).float().mean().item()
                    activation_stats[f"{name}@{call}"] = stats
                    return result
                handles.append(module.register_forward_pre_hook(observed_hook))
                a_names.append(name)
        # 始终保留全部平滑组，与 W/A 消融范围解耦，避免混入额外变量。
        save_json(args.output / "scope.json", {"smoothing_groups": count, "smoothed_inputs": sorted(smoothed),
            "fp32_pairs":fp32_pairs,"smoothing_bypassed":args.smoothing_bypass,
            "teacher_numerical_reference":"FP32_vision_pairs" if args.teacher_fp32_reference else "original_BF16",
                  "smoothing_selection": args.smoothing_selection,
                  "skipped_smoothing_groups": [g[0] for g in skipped],
                  "weight_targets": w_names, "activation_targets": a_names, "alpha": alpha})
    recorder = Recorder(model)
    from attention_probe import AttentionTap
    tap = AttentionTap(model, recorder, attention_layers) if attention_layers else None
    results = []
    try:
        for path in args.samples:
            activation_stats.clear()
            activation_calls.clear()
            reference = None if args.mode == "teacher" else load_trusted(args.teacher_dir / (path.stem + ".pt"))
            recorder.reset(reference["projector"] if args.oracle_projector and reference else None)
            if tap:
                tap.reset()
            raw, normalized, input_hashes = predict(path, cfg, model, head, proprio, processor)
            if args.export_candidate_teacher:
                torch.save({"raw":raw,"normalized":normalized,"traces":recorder.traces,
                    "attention":tap.data if tap else {},"projector":recorder.projector,"input_hashes":input_hashes},
                    args.output/(path.stem+".pt"))
            if recorder.projector is None:
                raise RuntimeError("Projector did not execute")
            if args.mode == "teacher":
                torch.save({"raw": raw, "normalized": normalized, "traces": recorder.traces,
                            "attention": tap.data if tap else {},
                            "projector": recorder.projector, "input_hashes": input_hashes}, args.output / (path.stem + ".pt"))
            else:
                if reference["input_hashes"] != input_hashes:
                    raise ValueError("Teacher/candidate preprocessed inputs differ")
                if set(reference["traces"]) != set(recorder.traces):
                    raise ValueError("Teacher/candidate trace keys differ")
                results.append({"sample": path.name,
                    "attention": tap.compare(reference["attention"]) if tap else {},
                    "normalized_action": metric(reference["normalized"], normalized),
                    "raw_action": metric(reference["raw"], raw),
                    "normalized_rmse_per_dim": (normalized - reference["normalized"]).square().mean(0).sqrt().tolist(),
                    "raw_rmse_per_dim": (raw - reference["raw"]).square().mean(0).sqrt().tolist(),
                    "normalized_rmse_per_step": (normalized - reference["normalized"]).square().mean(1).sqrt().tolist(),
                    # 与 rollout 的 sign(2*x-1) 相同，不能将连续值不相等当作开合分歧。
                    "raw_gripper_disagreement": (torch.sign(2 * raw[:, -1] - 1) != torch.sign(2 * reference["raw"][:, -1] - 1)).float().mean().item(),
                    "gripper_steps": {
                        "teacher_raw": reference["raw"][:, -1].tolist(),
                        "candidate_raw": raw[:, -1].tolist(),
                        "teacher_normalized": reference["normalized"][:, -1].tolist(),
                        "candidate_normalized": normalized[:, -1].tolist(),
                        "teacher_signed_margin": (2 * reference["raw"][:, -1] - 1).tolist(),
                        "candidate_signed_margin": (2 * raw[:, -1] - 1).tolist(),
                        "note": "按现有 rollout 的 sign(2*x-1) 记录；归一化值另行保留",
                    },
                    "activation_local_error": dict(activation_stats),
                    "features": {k: metric(reference["traces"][k], v) for k, v in recorder.traces.items()}})
                if len(results) == 1:
                    heatmaps = {}
                    for key, value in recorder.traces.items():
                        if key.startswith(("projector@", "llm_input@", "language_model.model.layers.31@")):
                            # 行表示抽样的 token 位置，不是完整图像网格。
                            delta = value.float() - reference["traces"][key].float()
                            heatmaps[key] = delta.squeeze(0).numpy()
                    np.savez_compressed(args.output / "sample0_feature_deltas.npz", **heatmaps)
                save_json(args.output / "metrics.json", results)
            print(f"[probe] {path.name}", flush=True)
    finally:
        if tap:
            tap.close()
        recorder.close()
        for handle in handles:
            handle.remove()
    print("PROBE COMPLETE (offline agreement only; not rollout success)", flush=True)


if __name__ == "__main__":
    main()
