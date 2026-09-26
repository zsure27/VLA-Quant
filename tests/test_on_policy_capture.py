import json
from pathlib import Path

import numpy as np

from qvla.on_policy_capture import OnPolicyObservationCapture, file_sha256


def test_capture_records_query_and_episode_outcome(tmpdir):
    tmp_path = Path(str(tmpdir))
    capture = OnPolicyObservationCapture(tmp_path, chunk_execution_steps=8)
    observation = {
        "full_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
    }
    capture.begin_episode("pick up the bowl")
    capture.record_query(
        observation,
        "pick up the bowl",
        np.zeros((8, 7), dtype=np.float32),
        {"h17_token_mean": np.zeros(3), "h17_last_token": np.ones(3)},
    )
    capture.end_episode(True)
    capture.close()

    events = [json.loads(line) for line in (tmp_path / "on-policy-events.jsonl").read_text().splitlines()]
    assert [row["record_type"] for row in events] == ["episode_start", "query", "episode_end"]
    assert events[1]["executed_step_start"] == 0
    assert events[2]["success"] is True
    sample = tmp_path / events[1]["file"]
    assert file_sha256(sample) == events[1]["file_sha256"]
    with np.load(sample, allow_pickle=False) as payload:
        assert payload["student_action"].shape == (8, 7)
        assert payload["h17_token_mean"].shape == (3,)
