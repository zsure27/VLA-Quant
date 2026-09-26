import numpy as np

from qvla.on_policy_capture import action_metrics


def test_action_metrics_split_components_and_gripper_threshold():
    teacher = np.zeros((8, 7), dtype=np.float32)
    student = teacher.copy()
    student[:, 0] = 1.0
    student[:, 3] = 2.0
    student[:, 6] = 1.0
    values = action_metrics(student, teacher)
    assert values["position_rmse"] == np.sqrt(1 / 3)
    assert values["rotation_rmse"] == np.sqrt(4 / 3)
    assert values["gripper_continuous_rmse"] == 1.0
    assert values["gripper_disagreement"] == 1.0
    assert len(values["rmse_per_step"]) == 8
