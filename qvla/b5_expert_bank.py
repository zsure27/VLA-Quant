"""Frozen B3/B4 residual bank on one already-validated A4 quantized model.

This module does not quantize a model and does not relax archived loaders. The
outer B5 artifact contract must pin the A4 checkpoint/profile and expert files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from qvla.b5_route_state import ChunkRouteState


BANK_VERSION = "b5-shared-a4-frozen-b3-b4-bank-v1"
SUFFIX = ".awq_recovery_lora."


class FrozenExpertBank(torch.nn.Module):
    def __init__(self, model, route_state: ChunkRouteState, b3_state: dict, b4_state: dict,
                 *, language_targets=None, visual_targets=None, canonical: bool = True):
        super().__init__()
        if not isinstance(route_state, ChunkRouteState):
            raise TypeError("Expert bank requires the strict chunk route state")
        if canonical:
            from qvla.extended_peft import target_names, validate_shapes
            from qvla.b4_joint_peft import validate_shapes as validate_joint
            expected_language, expected_visual = target_names("B3"), target_names("B2")
            if language_targets is not None and set(language_targets) != expected_language:
                raise ValueError("B5 language coverage differs from canonical B3")
            if visual_targets is not None and set(visual_targets) != expected_visual:
                raise ValueError("B5 visual coverage differs from canonical B4")
            language_targets, visual_targets = expected_language, expected_visual
            validate_shapes({k: tuple(v.shape) for k, v in b3_state.items()}, "B3")
            validate_joint({k: tuple(v.shape) for k, v in b4_state.items()})
        elif language_targets is None or visual_targets is None:
            raise ValueError("Noncanonical toy bank requires explicit target sets")
        self.language_targets = tuple(sorted(language_targets))
        self.visual_targets = tuple(sorted(visual_targets))
        if set(self.language_targets) & set(self.visual_targets):
            raise ValueError("Language and vision targets overlap")
        self.target_names = tuple(sorted(set(self.language_targets) | set(self.visual_targets)))
        self.route_state = route_state
        self.canonical = canonical
        self.branches = torch.nn.ModuleDict()
        self._handles, self._targets = [], []
        modules = dict(model.named_modules())
        states = (b3_state, b4_state)
        for expert, names in enumerate((self.language_targets, self.target_names)):
            expected_keys = {f"{n}{SUFFIX}{i}.weight" for n in names for i in (0, 1)}
            if set(states[expert]) != expected_keys:
                raise ValueError("Expert adapter coverage mismatch")
        # Complete validation before mutating any source module.
        for name in self.target_names:
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"Bank target is not a Linear: {name}")
            if any(hasattr(module, attr) for attr in ("awq_recovery_lora", "awq_scale_peft", "_b5_bank_attached")):
                raise ValueError(f"Target already has an adapter or bank: {name}")
            for expert, state in enumerate(states):
                if expert == 0 and name in self.visual_targets:
                    continue
                down, up = (state[f"{name}{SUFFIX}{i}.weight"] for i in (0, 1))
                if any(not isinstance(v, torch.Tensor) or not v.is_floating_point() or
                       not torch.isfinite(v).all() for v in (down, up)):
                    raise ValueError("Expert state must contain finite floating tensors")
                if (down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or
                        down.shape[1] != module.in_features or up.shape != (module.out_features, down.shape[0])):
                    raise ValueError(f"Adapter dimensions disagree with base: {name}")
        model.eval()
        model.requires_grad_(False)
        for index, name in enumerate(self.target_names):
            module = modules[name]
            for expert, state in enumerate(states):
                if expert == 0 and name in self.visual_targets:
                    continue
                down, up = (state[f"{name}{SUFFIX}{i}.weight"] for i in (0, 1))
                branch = torch.nn.Sequential(torch.nn.Linear(module.in_features, down.shape[0], bias=False),
                                             torch.nn.Linear(down.shape[0], module.out_features, bias=False))
                branch.to(device=module.weight.device, dtype=module.weight.dtype)
                with torch.no_grad():
                    branch[0].weight.copy_(down.to(branch[0].weight))
                    branch[1].weight.copy_(up.to(branch[1].weight))
                branch.requires_grad_(False)
                self.branches[self._key(index, expert)] = branch
        self.requires_grad_(False)
        model.requires_grad_(False)
        model.eval()
        self.eval()
        try:
            for index, name in enumerate(self.target_names):
                module = modules[name]
                module._b5_bank_attached = True
                self._targets.append(module)
                self._handles.append(module.register_forward_hook(self._hook(index)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _key(index: int, expert: int) -> str:
        return f"target_{index}_expert_{expert}"

    def _hook(self, index):
        def correction(_module, args, output):
            expert = self.route_state.active_expert
            key = self._key(index, expert)
            # E0 vision must exactly follow the bare B3 base path.
            return output if key not in self.branches else output + self.branches[key](args[0])
        return correction

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def tensor_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def export_expert_states(self):
        states = ({}, {})
        for index, name in enumerate(self.target_names):
            for expert in (0, 1):
                key = self._key(index, expert)
                if key in self.branches:
                    for i in (0, 1):
                        states[expert][f"{name}{SUFFIX}{i}.weight"] = self.branches[key][i].weight.detach().cpu().clone()
        return states

    def frozen_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path, *, provenance: dict) -> None:
        if self.route_state.active_token is not None:
            raise RuntimeError("Cannot serialize during an expert forward")
        e0, e1 = self.export_expert_states()
        payload = {"version": BANK_VERSION, "base_model_id": "A4", "canonical": self.canonical,
                   "language_targets": list(self.language_targets), "visual_targets": list(self.visual_targets),
                   "provenance": provenance, "E0": e0, "E1": e1}
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path, model, route_state, *, expected_provenance: dict,
             canonical: bool = True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected_keys = {"version", "base_model_id", "canonical", "language_targets", "visual_targets", "provenance", "E0", "E1"}
        if (not isinstance(payload, dict) or set(payload) != expected_keys or
                payload["version"] != BANK_VERSION or payload["base_model_id"] != "A4" or
                payload["canonical"] != canonical or payload["provenance"] != expected_provenance):
            raise ValueError("Saved expert bank identity/provenance mismatch")
        return cls(model, route_state, payload["E0"], payload["E1"],
                   language_targets=payload["language_targets"], visual_targets=payload["visual_targets"],
                   canonical=canonical)

    def close(self):
        for handle in self._handles:
            handle.remove()
        for target in self._targets:
            if hasattr(target, "_b5_bank_attached"):
                delattr(target, "_b5_bank_attached")
        self._handles, self._targets = [], []
