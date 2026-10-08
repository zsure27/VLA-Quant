"""Explicit state-space handling for captured on-policy PEFT observations.

The AWQ profile fingerprints action_jacobian_batch.py. Keep that module byte
identical and put the new training-only contract here.
"""
from __future__ import annotations

import numpy as np


def prepare_proprio(state, stats, state_space: str):
    if state_space == "raw_proprio_before_get_action":
        from experiments.robot.openvla_utils import normalize_proprio
        return normalize_proprio(state.copy(), stats)
    if state_space == "policy_normalized_proprio":
        result = state.copy()
        if result.shape != (8,) or not np.isfinite(result).all() or np.any(np.abs(result) > 1.000001):
            raise ValueError("Captured policy-normalized proprio is malformed")
        return result
    raise ValueError(f"Unknown proprio state space: {state_space}")


def prepare_policy_inputs(sample, cfg, model, processor, device):
    from qvla.action_jacobian_batch import prepare_inputs

    # Reuse the profile-fingerprinted image/token preparation. Its proprio
    # output is deliberately discarded: this sample is already normalized by
    # the deployed OFT policy before on-policy capture wrote the NPZ.
    inputs, _discarded_state = prepare_inputs(sample, cfg, model, processor, device)
    stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    return inputs, prepare_proprio(sample["state"], stats, "policy_normalized_proprio")
