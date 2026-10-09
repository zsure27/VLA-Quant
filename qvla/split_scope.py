"""Separate episode generalization from Recovery-stage task transfer (CPU only)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / 'results/experiments/p2-shared-peft/20260930-014-a1-reset-preflight/unpruned_state_inventory.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def task_inventory():
    rows = json.loads(INVENTORY.read_text(encoding='utf-8'))['tasks']
    return {r['task_id']: r for r in rows}


def audit_scope(training_rows, evaluation_rows, claim='within_task_development',
                calibration_task_ids=None, initialization_task_ids=None):
    """Require content evidence, not just different reset indices or filenames."""
    train_keys = {(r['task_id'], r['init_state_index']) for r in training_rows}
    eval_keys = [(r['task_id'], r['init_state_index']) for r in evaluation_rows]
    if not train_keys or not eval_keys or len(set(eval_keys)) != len(eval_keys):
        raise ValueError('Empty cohort or duplicate evaluation episode')
    if train_keys & set(eval_keys): raise ValueError('Training/evaluation episode overlap')
    train_init = {r['init_state_sha256'] for r in training_rows}
    eval_init = {r['init_state_sha256'] for r in evaluation_rows}
    if not all(train_init | eval_init): raise ValueError('Missing actual initial-state SHA')
    if train_init & eval_init: raise ValueError('Training/evaluation initial-state content overlap')
    train_obs = {r['observation_sha256'] for r in training_rows}
    eval_obs = {h for r in evaluation_rows for h in r['observation_sha256_all_queries']}
    if any(not r['observation_sha256_all_queries'] for r in evaluation_rows):
        raise ValueError('Missing observations for an evaluation episode')
    if not train_obs or not eval_obs or not all(train_obs | eval_obs):
        raise ValueError('Missing policy-visible observation content evidence')
    if train_obs & eval_obs: raise ValueError('Training/evaluation observation content overlap')
    train_tasks = {r['task_id'] for r in training_rows}
    eval_tasks = {r['task_id'] for r in evaluation_rows}
    if claim == 'recovery_task_transfer_diagnostic':
        if train_tasks & eval_tasks: raise ValueError('Task-transfer claim contains trained tasks')
        if calibration_task_ids is None or initialization_task_ids is None:
            raise ValueError('Task-transfer requires calibration and initialization exposure manifests')
        if eval_tasks & (set(calibration_task_ids) | set(initialization_task_ids)):
            raise ValueError('Held-out Recovery task exposed to calibration or initialization')
    elif claim != 'within_task_development':
        raise ValueError('Final/unseen-foundation-task claims are not admitted by this contract')
    return {'gate': 'PASS_DATA_SCOPE', 'claim': claim,
            'training_tasks': sorted(train_tasks), 'evaluation_tasks': sorted(eval_tasks),
            'shared_tasks': sorted(train_tasks & eval_tasks), 'training_episodes': len(train_keys),
            'evaluation_episodes': len(eval_keys), 'initial_state_overlap': 0, 'observation_overlap': 0,
            'foundation_training_exclusion': 'NOT_ESTABLISHED',
            'final_evaluation': False}
