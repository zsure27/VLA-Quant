"""Admit temporal coverage training without conflating different teacher inputs."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--steps', type=int, choices=(10, 1000), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, os.environ['VLA_FROZEN_SOURCE'])
    import torch
    from qvla.extended_peft import sha
    records, initials, teachers = {}, {}, {}
    for variant in ('early80', 'phase80'):
        root = a.root / variant / ('steps-%d' % a.steps)
        rec = json.loads((root / 'artifact.json').read_text())
        if (rec['version'] != 'b1-temporal80-v1-20261017' or rec['temporal_variant'] != variant
                or rec['steps'] != a.steps or rec['initialization'] != 'response-svd'
                or rec['gate'] != ('PASS_SMOKE10' if a.steps == 10 else 'PASS_TRAIN_CONTRACT')
                or rec['parameters'] != 19988480 or not rec['zero_output_equal'] or not rec['reload_output_equal']
                or rec['frozen_before_sha256'] != rec['frozen_after_sha256']):
            raise ValueError('Temporal numerical/gradient/frozen gate failed')
        for name, digest in rec['reuse_files'].items():
            if sha(root / name) != digest:
                raise ValueError('Temporal saved material changed')
        for name, digest in rec['source_sha256'].items():
            # Relative dependencies are resolved only within the old frozen source.
            f = Path(name) if Path(name).is_absolute() else Path(os.environ['VLA_FROZEN_SOURCE']) / name
            if sha(f) != digest:
                raise ValueError('Actual source changed')
        initials[variant] = torch.load(root / 'initial-adapter.pt', map_location='cpu', weights_only=True)
        teachers[variant] = torch.load(root / 'teacher-targets.pt', map_location='cpu', weights_only=True)
        if set(teachers[variant]) != {'sample-%05d.npz' % i for i in range(80)}:
            raise ValueError('Teacher coverage mismatch')
        if any(t.shape != (8, 7) or not torch.isfinite(t).all() for t in teachers[variant].values()):
            raise ValueError('Invalid teacher action')
        records[variant] = rec
    early, phase = records.values()
    for field in ('checkpoint_identity', 'targets', 'target_sha256', 'source_sha256', 'spec_sha256',
                  'sample_sequence_sha256', 'frozen_before_sha256', 'inherited_initial_factor_sha256'):
        if early[field] != phase[field]:
            raise ValueError('Uncontrolled temporal comparison difference: ' + field)
    if early['training_manifest_sha256'] == phase['training_manifest_sha256']:
        raise ValueError('Temporal cohorts unexpectedly identical')
    x, y = initials.values()
    if set(x) != set(y) or any(not torch.equal(x[k], y[k]) for k in x):
        raise ValueError('Temporal cohorts did not start from identical SVD factors')
    result = {'gate': 'PASS_B1_TEMPORAL_SMOKE_PAIR' if a.steps == 10 else 'PASS_B1_TEMPORAL_TRAIN_PAIR',
              'steps': a.steps, 'parameters_each': 19988480, 'same_initial_tensor_values': True,
              'same_training_order_slots': True, 'teacher_labels_scope': 'Separate fresh BF16 labels for each cohort; Early80 all80 bridges parent cache',
              'artifacts': {v: sha(a.root / v / ('steps-%d' % a.steps) / 'artifact.json') for v in records},
              'scope': 'Structural training gate only, no closed-loop claim'}
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
