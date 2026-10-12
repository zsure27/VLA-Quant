"""CPU contracts; these do not claim real-model or GPU admission."""
import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from qvla.b5_route_state import ChunkRouteState
from qvla.b5_expert_bank import FrozenExpertBank
from qvla.b5_features import CausalFeatureEncoder, RAW_STATE, NORMALIZED_STATE, normalized_proprio
from qvla.extended_peft import attach_state


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.vision = torch.nn.Linear(3, 4, dtype=dtype)
        self.language = torch.nn.Linear(4, 2, dtype=dtype)

    def forward(self, value):
        return self.language(torch.tanh(self.vision(value)))


def materials(dtype=torch.float32):
    torch.manual_seed(8)
    base = Toy(dtype)
    state0, state1 = {}, {}
    for index, names in enumerate((("language",), ("language", "vision"))):
        state = state0 if index == 0 else state1
        for name in names:
            module = getattr(base, name)
            state[f"{name}.awq_recovery_lora.0.weight"] = torch.randn(2, module.in_features, dtype=dtype)
            state[f"{name}.awq_recovery_lora.1.weight"] = torch.randn(module.out_features, 2, dtype=dtype)
    return base, state0, state1


def bank(base, route, state0, state1):
    return FrozenExpertBank(base, route, state0, state1, canonical=False,
                            language_targets={"language"}, visual_targets={"vision"})


def query(model, route, value, expert, index=0):
    token = route.begin_query("episode", index, expert, executed_step_start=index * 8)
    try:
        result = model(value)
    except BaseException:
        route.abort_query(token)
        raise
    route.end_query(token)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_static_parity_switching_frozen_and_shared_base(dtype):
    base, e0, e1 = materials(dtype)
    references = [copy.deepcopy(base), copy.deepcopy(base)]
    for reference, state in zip(references, (e0, e1)):
        attach_state(reference, state)
    original = {name: value.clone() for name, value in base.state_dict().items()}
    route = ChunkRouteState()
    dynamic = bank(base, route, e0, e1)
    before = dynamic.frozen_sha256()
    assert all(not p.requires_grad for p in dynamic.parameters())
    assert all(not p.requires_grad for p in base.parameters())
    assert not any("base" in name for name in dynamic.state_dict())
    route.begin_episode("episode")
    value = torch.randn(2, 5, 3, dtype=dtype)
    for index, expert in enumerate((0, 1, 0)):
        actual = query(base, route, value, expert, index)
        assert torch.equal(actual, references[expert](value))
        for _ in range(8):
            route.record_executed_step()
    assert before == dynamic.frozen_sha256()
    assert all(torch.equal(value, base.state_dict()[name]) for name, value in original.items())
    route.end_episode()
    with pytest.raises(RuntimeError):
        base(value)
    dynamic.close()
    assert all(torch.equal(value, base.state_dict()[name]) for name, value in original.items())


def test_save_reload_provenance_and_duplicate_guard():
    base, e0, e1 = materials()
    fresh = copy.deepcopy(base)
    route = ChunkRouteState()
    dynamic = bank(base, route, e0, e1)
    with pytest.raises(ValueError):
        bank(base, route, e0, e1)
    parent = Path.cwd() / ".local" / "b5-cpu-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        path = Path(directory) / "bank.pt"
        provenance = {"base_profile_sha256": "a" * 64, "adapter0_sha256": "b" * 64, "adapter1_sha256": "c" * 64}
        dynamic.save(path, provenance=provenance)
        with pytest.raises(FileExistsError):
            dynamic.save(path, provenance=provenance)
        with pytest.raises(ValueError):
            FrozenExpertBank.load(path, fresh, ChunkRouteState(), expected_provenance={}, canonical=False)
        reloaded_route = ChunkRouteState()
        reloaded = FrozenExpertBank.load(path, fresh, reloaded_route, expected_provenance=provenance, canonical=False)
        assert dynamic.frozen_sha256() == reloaded.frozen_sha256()
        value = torch.randn(3, 3)
        for expert in (0, 1):
            route.begin_episode("episode")
            reloaded_route.begin_episode("episode")
            assert torch.equal(query(base, route, value, expert), query(fresh, reloaded_route, value, expert))
            route.end_episode()
            reloaded_route.end_episode()
        dynamic.close()
        reloaded.close()


def test_static_adapter_and_malformed_state_rejected_before_mutation():
    base, e0, e1 = materials()
    attach_state(base, e0)
    with pytest.raises(ValueError):
        bank(base, ChunkRouteState(), e0, e1)
    base, e0, e1 = materials()
    e1["vision.awq_recovery_lora.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        bank(base, ChunkRouteState(), e0, e1)
    assert not hasattr(base.language, "_b5_bank_attached")


def test_query_lifecycle_exact_chunk_abort_and_early_stop():
    route = ChunkRouteState()
    with pytest.raises(RuntimeError):
        _ = route.active_expert
    route.begin_episode("episode")
    token = route.begin_query("episode", 0, 0)
    with pytest.raises(RuntimeError):
        route.begin_query("episode", 0, 1)
    with pytest.raises(ValueError):
        route.end_query(token, action_count=7)
    route.end_query(token)
    for _ in range(7):
        route.record_executed_step()
    with pytest.raises(RuntimeError):
        route.begin_query("episode", 1, 1)
    route.record_executed_step()
    token2 = route.begin_query("episode", 1, 1, executed_step_start=8)
    with pytest.raises(RuntimeError):
        route.end_query(token)
    route.abort_query(token2)
    with pytest.raises(RuntimeError):
        route.begin_query("episode", 1, 0)
    assert route.end_episode()["aborted"]
    route.begin_episode("episode")
    token3 = route.begin_query("episode", 0, 0)
    route.end_query(token3)
    route.record_executed_step()
    receipt = route.end_episode()
    assert receipt["queries"] == 1 and receipt["discarded_actions"] == 7


def test_causal_features_schema_and_explicit_normalization():
    encoder = CausalFeatureEncoder([f"task {i}" for i in range(10)])
    obs = {"full_image": np.zeros((17, 25, 3), np.uint8),
           "wrist_image": np.full((17, 25, 3), 255, np.uint8), "state": np.zeros(8)}
    features = encoder.encode(obs, "TASK  3", state_space=NORMALIZED_STATE)
    assert features.shape == (402,) and features.dtype == np.float32
    assert np.all(features[:192] == 0) and np.all(features[192:384] == 1)
    assert features[392:].sum() == 1 and features[395] == 1
    rebuilt = CausalFeatureEncoder.from_schema(encoder.schema())
    assert rebuilt.schema_sha256 == encoder.schema_sha256
    assert np.array_equal(rebuilt.encode(obs, "task 3", state_space=NORMALIZED_STATE), features)
    for forbidden in ("success", "h17", "teacher_actions", "init_state_index", "future_observation"):
        with pytest.raises(ValueError):
            encoder.encode(dict(obs, **{forbidden: 1}), "task 3", state_space=NORMALIZED_STATE)
    stats = {"q01": [-2.] * 8, "q99": [2.] * 8, "mask": [True] * 8}
    state = np.arange(8, dtype=np.float64) - 3
    expected = np.clip(2 * (state + 2) / (4 + 1e-8) - 1, -1, 1).astype(np.float32)
    assert np.array_equal(normalized_proprio(state, state_space=RAW_STATE, norm_stats=stats), expected)
    with pytest.raises(ValueError):
        normalized_proprio(np.zeros(8), state_space=NORMALIZED_STATE, norm_stats=stats)
    with pytest.raises(ValueError):
        encoder.encode(obs, "unregistered task", state_space=NORMALIZED_STATE)
