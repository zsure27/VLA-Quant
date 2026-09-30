from pathlib import Path


def test_paired_evaluator_applies_environment_seed_after_global_model_seed():
    helper = Path(__file__).parents[1] / "overlays/openvla-oft/experiments/robot/libero/run_libero_eval.py"
    source = helper.read_text(encoding="utf-8")
    block = source.split('if cfg.seed_protocol == "paired":', 1)[1].split("import hashlib", 1)[0]

    assert block.index("seed_all(model_seed") < block.index("env.seed(environment_seed)")
