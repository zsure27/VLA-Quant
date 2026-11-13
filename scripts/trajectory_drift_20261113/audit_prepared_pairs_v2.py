"""Original raw audit/geometry; only profile identity dispatch extends old math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--series', type=Path, required=True)
    p.add_argument('--round-dir', type=Path)
    p.add_argument('--existing-component-check', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = a.root.resolve()
    sys.path[:0] = [str(r / 'code-cpu-prep-20261109-v1'), str(r / 'sources/openvla-oft'), str(r / 'sources/LIBERO')]
    series = json.loads(a.series.read_text())
    source = r / 'code-cpu-prep-20261109-v1/diagnostics/native_oft_trace_v1.py'
    expected = 'b084a872cccbeb839ac83bc74cc88fb93ce0cfa841d616b72addc21b498a576f'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    body = source.read_text()
    old = "profile = 'awq.pt' if m['condition'].startswith('awq') else 'smoothquant.pt'\n        assert result['profile_sha256'] == sha(Path('profiles/20261109-four-suite-reset0-v4')/profile)"
    assert body.count(old) == 1
    body = body.replace(old, "assert result['profile_sha256'] == expected_profiles[m['condition']]")
    m = types.ModuleType('_original_raw_audit_profile_dispatch')
    m.expected_profiles = {k: v['sha256'] for k, v in series['profiles'].items()}
    exec(compile(body, str(source) + '[profile identity dispatch only]', 'exec'), m.__dict__)
    cache = {}
    original_sha = m.sha
    def cached_sha(path):
        path = Path(path)
        key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        if key not in cache:
            cache[key] = original_sha(path)
        return cache[key]
    m.sha = cached_sha
    if a.existing_component_check:
        summary = json.loads((r / 'control/20261112-061-oft-qvla-resume-v6/analysis-oft-seven-conditions-v2/summary.json').read_text())
        cases = [x for x in summary['metrics'] if x['task_id'] == 0 and x['reset_index'] == 5]
        assert len(cases) == 24
        rows = [dict(case=dict(condition=x['condition']), result_path=str(Path(x['candidate']) / 'result.json'),
                     reference_result=str(Path(x['reference']) / 'result.json')) for x in cases]
    else:
        assert a.round_dir and (a.round_dir / 'series.json').resolve() == a.series.resolve()
        rows = []
        for f in a.round_dir.glob('*-libero*-cases.json'):
            for row in json.loads(f.read_text())['completed']:
                assert cached_sha(row['result_path']) == row['result_sha256']
                rows.append(dict(row, reference_result=row['case']['reference_result']))
        keys = [(x['case']['condition'], x['case']['suite'], x['case']['task_id'], x['case']['reset_index']) for x in rows]
        assert len(keys) == len(set(keys))
        if len(rows) != len(series['cases']):
            assert (a.round_dir / 'STOP_NEW_GPU').exists(), 'Unexpected missing cases, not a quota stop'
    metrics = []
    for row in rows:
        left, right = m.load(Path(row['result_path']).parent), m.load(Path(row['reference_result']).parent)
        position, rotation, _ = m.compare(left, right)
        condition = row['case']['condition']
        assert left['result']['condition'] == condition
        assert left['result']['profile_sha256'] == m.expected_profiles[condition]
        assert np.isfinite(position).all() and np.isfinite(rotation).all()
        metrics.append(dict(condition=condition, candidate=row['result_path'], reference=row['reference_result'],
                            common_steps=len(position), position_p95_mm=float(np.percentile(position, 95) * 1000),
                            rotation_p95_deg=float(np.percentile(rotation, 95)),
                            success=left['result']['success'], reference_success=right['result']['success']))
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(gate='PASS_EXISTING24_RAW_PAIRS_CPU_NEW_AUDIT_DISPATCH' if a.existing_component_check
                  else 'PASS_PREPARED_CONTINUATION_AVAILABLE_RAW_PAIRED_CASES',
                  completed=len(metrics), qualification='24 existing pairs only; zero new rollouts' if a.existing_component_check
                  else 'Partial only allowed under STOP_NEW_GPU; use actual completed coverage',
                  raw_geometry_sha256=expected, profile_dispatch_only_changed=True,
                  metrics=metrics, raw_audits=m.AUDITS)
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['metrics', 'raw_audits']}), flush=True)


if __name__ == '__main__':
    main()
