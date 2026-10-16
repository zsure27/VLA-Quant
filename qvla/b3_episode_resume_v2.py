"""Scheduling-only adapter for the three interrupted M0 episodes."""
from __future__ import annotations
import copy
import inspect

MISSING = ((9, 7), (9, 8), (9, 9))


def validate_partition(completed, missing):
    expected = [(t, r) for t in range(10) for r in range(5, 10)]
    if list(completed) != expected[:47] or tuple(missing) != MISSING:
        raise ValueError('Only the verified 47+3 recovery boundary is registered')
    if set(completed) & set(missing) or list(completed) + list(missing) != expected:
        raise ValueError('Overlap or incomplete episode coverage')


def selected_run_task(original, visited):
    signature = inspect.signature(original)

    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        task = bound.arguments['task_id']
        if task != 9:
            return bound.arguments['total_episodes'], bound.arguments['total_successes']
        if visited:
            raise ValueError('Selected task would repeat')
        cfg = copy.copy(bound.arguments['cfg'])
        if (cfg.initial_state_offset, cfg.num_trials_per_task, cfg.seed_protocol) != (7, 3, 'paired'):
            raise ValueError('Missing episodes must be exactly task9 reset7-9 under paired protocol')
        bound.arguments['cfg'] = cfg
        visited.extend(MISSING)
        return original(*bound.args, **bound.kwargs)

    return run
