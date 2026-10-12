"""Independent causal image/proprio/instruction features, before VLA forward.

The encoder deliberately has no access to a model, adapter, teacher or outcome.
Its fixed 402-dimensional schema is suitable for a small B5 pilot classifier.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np


FEATURE_VERSION = "b5-causal-grid8-proprio8-instruction10-v1"
FEATURE_DIM = 402
OBSERVATION_KEYS = frozenset(("full_image", "wrist_image", "state"))
RAW_STATE = "raw_proprio_before_get_action"
NORMALIZED_STATE = "policy_normalized_proprio"


def canonical_instruction(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Instruction must be a nonempty string")
    return " ".join(text.lower().split())


def _vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite eight-dimensional vector")
    return result


def normalized_proprio(state, *, state_space: str, norm_stats=None,
                       normalization_type: str = "BOUNDS_Q99") -> np.ndarray:
    value = _vector(state, "proprio").copy()
    if state_space == NORMALIZED_STATE:
        if norm_stats is not None:
            raise ValueError("Already-normalized proprio must not be normalized again")
        if np.any(np.abs(value) > 1.000001):
            raise ValueError("Normalized proprio is outside the policy bounds")
        return value.astype(np.float32)
    if state_space != RAW_STATE or not isinstance(norm_stats, Mapping):
        raise ValueError("Raw proprio requires explicit policy normalization statistics")
    if normalization_type == "BOUNDS_Q99":
        low, high = _vector(norm_stats["q01"], "q01"), _vector(norm_stats["q99"], "q99")
    elif normalization_type == "BOUNDS":
        low, high = _vector(norm_stats["min"], "min"), _vector(norm_stats["max"], "max")
    else:
        raise ValueError("Unknown policy proprio normalization type")
    mask = np.asarray(norm_stats.get("mask", [True] * 8))
    if mask.shape != (8,) or mask.dtype != np.bool_ or np.any((high < low) & mask):
        raise ValueError("Malformed proprio bounds or normalization mask")
    # Match the deployed OFT formula, including epsilon and mask semantics.
    result = np.clip(np.where(mask, 2 * (value - low) / (high - low + 1e-8) - 1, value), -1, 1)
    return result.astype(np.float32)


def _grid(image) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 8:
        raise ValueError("Causal camera must be uint8 HxWx3 with H,W >= 8")
    height, width = value.shape[:2]
    result = np.empty((8, 8, 3), dtype=np.float64)
    for row in range(8):
        for col in range(8):
            tile = value[row * height // 8:(row + 1) * height // 8,
                         col * width // 8:(col + 1) * width // 8]
            result[row, col] = tile.mean(axis=(0, 1), dtype=np.float64) / 255.0
    return result.reshape(-1).astype(np.float32)


class CausalFeatureEncoder:
    def __init__(self, instruction_vocabulary: Sequence[str]):
        vocabulary = tuple(canonical_instruction(text) for text in instruction_vocabulary)
        if len(vocabulary) != 10 or len(set(vocabulary)) != 10:
            raise ValueError("Spatial pilot needs an exact, ordered ten-instruction vocabulary")
        self.instruction_vocabulary = vocabulary

    def schema(self) -> dict:
        return {"version": FEATURE_VERSION, "dimension": FEATURE_DIM,
                "input_keys": sorted(OBSERVATION_KEYS), "grid": [8, 8, 3],
                "feature_order": ["full_image", "wrist_image", "normalized_proprio", "instruction_one_hot"],
                "image_preparation": "mean_grid_of_pre_get_action_camera_uint8",
                "instruction_vocabulary": list(self.instruction_vocabulary)}

    @property
    def schema_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.schema(), sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_schema(cls, schema: Mapping) -> "CausalFeatureEncoder":
        encoder = cls(schema["instruction_vocabulary"])
        if dict(schema) != encoder.schema():
            raise ValueError("B5 feature schema differs from the registered encoder")
        return encoder

    def encode(self, observation: Mapping, instruction: str, *, state_space: str,
               norm_stats=None, normalization_type: str = "BOUNDS_Q99") -> np.ndarray:
        if not isinstance(observation, Mapping) or set(observation) != OBSERVATION_KEYS:
            raise ValueError("Router observation must contain only cameras and current proprio")
        text = canonical_instruction(instruction)
        if text not in self.instruction_vocabulary:
            raise ValueError("Instruction is outside the registered Spatial mapping")
        instruction_feature = np.zeros(10, dtype=np.float32)
        instruction_feature[self.instruction_vocabulary.index(text)] = 1
        features = np.concatenate((_grid(observation["full_image"]), _grid(observation["wrist_image"]),
                    normalized_proprio(observation["state"], state_space=state_space,
                        norm_stats=norm_stats, normalization_type=normalization_type), instruction_feature))
        if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
            raise RuntimeError("Invalid B5 causal features")
        return features

    def task_only(self, instruction: str) -> np.ndarray:
        """Same instruction mapping for the explicitly separate task-only control."""
        text = canonical_instruction(instruction)
        if text not in self.instruction_vocabulary:
            raise ValueError("Unknown task-only instruction")
        value = np.zeros(10, dtype=np.float32)
        value[self.instruction_vocabulary.index(text)] = 1
        return value
