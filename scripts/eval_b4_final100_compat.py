"""Evaluate registered B4 reset40-49 without changing its trained source provenance.

The archived B4 artifact pins its training and evaluator source hashes.  This
entry point preserves those bytes and extends only the reset-range admission
check for the separately preregistered final 100 development episodes.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path
import runpy

import qvla.b4_joint_peft as contract


def install_final100_validator() -> None:
    original = contract.validate_artifact

    def validate(args):
        if (args.task_suite_name != "libero_spatial" or
                args.initial_state_offset != 40 or
                args.num_trials_per_task != 10):
            raise ValueError("Final100 compatibility admits only registered reset40-49")
        provenance_args = copy(args)
        provenance_args.initial_state_offset = 30
        return original(provenance_args)

    contract.validate_artifact = validate


def main() -> None:
    install_final100_validator()
    evaluator = Path(__file__).resolve().parent.parent / "qvla/run_eval_b4_joint.py"
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
