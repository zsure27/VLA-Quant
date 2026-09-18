"""Assemble three-file review from committed measurements; no server/network."""
import csv
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / 'reports/2026-09-16-weekend-review'
ARCHIVE = REPO / 'reports/2026-09-16-weekend-review-archive'
ASSETS = REPO / 'reports/assets/20260918-weekend-review'
LATEST = REPO / 'reports/sessions/20260918-107-awq-continuation'
BASELINE = REPO / 'reports/sessions/20260917-107-baseline-continuation'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def relative(path):
    return path.relative_to(REPO).as_posix()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compact_raw_metrics(value):
    """Index large numeric diagnostic arrays; retain original files and scalars."""
    if isinstance(value, dict):
        output = {}
        for key, field in value.items():
            encoded = json.dumps(field, separators=(',', ':'), allow_nan=False).encode('utf-8')
            if key in ['features', 'attention', 'activation_local_error'] and len(encoded) > 2048:
                output[key] = {'indexed_diagnostic_block': True, 'canonical_json_bytes': len(encoded),
                               'canonical_json_sha256': hashlib.sha256(encoded).hexdigest(),
                               'recovery': 'Read this field from the SHA256-indexed original source file'}
            else: output[key] = compact_raw_metrics(field)
        return output
    if isinstance(value, list):
        if len(value) > 64 and all(isinstance(x, (int, float)) for x in value):
            return {'indexed_numeric_array': True, 'length': len(value),
                    'canonical_json_sha256': hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest(),
                    'recovery': 'Read this field from the SHA256-indexed original source file'}
        return [compact_raw_metrics(v) for v in value]
    return value

def build():
    baseline = read_json(BASELINE / 'data/summary.json')
    scope = read_json(LATEST / 'data/scope_summary.json')
    lookup = {row['case']: row for row in scope}
    assert baseline['paired_episodes'] == 500
    assert [(r['case'], r['successes']) for r in baseline['variants']] == [('bf16', 487), ('w4', 486)]
    assert lookup['vision-w2']['episodes'] == 500 and lookup['vision-w2']['successes'] == 460
    assert lookup['full-w2-original']['episodes'] == 50 and lookup['full-w2-original']['successes'] == 0
    assert lookup['full-w2-g64-attention-no-clip']['episodes'] == 50 and lookup['full-w2-g64-attention-no-clip']['successes'] == 11
    splits = read_json(LATEST / 'data/language_attention_splits.json')
    assert [(r['episodes'], r['successes']) for r in splits] == [(50, 29), (100, 49)]
    factorial = read_json(LATEST / 'data/language_clip_factorial.json')
    assert factorial['matrix'] == [[0, 0], [29, 20]]
    pair_path = LATEST / 'data/scope_paired_episodes.csv'
    with pair_path.open(encoding='utf-8-sig', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    keys = [(r['case'], r['task_id'], r['init_state_index']) for r in pairs]
    assert len(keys) == len(set(keys)), 'Duplicate case/task/init in latest scope data'
    with (BASELINE / 'data/paired_episodes.csv').open(encoding='utf-8-sig', newline='') as stream:
        formal_pairs = list(csv.DictReader(stream))
    assert len(formal_pairs) == 500
    # Scientific figures use measured counts, not synthetic samples.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    panels = [(['BF16', 'AWQ W4', 'Vision W2'], [487, 486, 460], 500, 'Spatial500: official states 0-49'),
              (['BF16', 'Full W2 G128', 'Full W2 G64\nattention no clip', 'Language W2 G64\nattention no clip'], [50, 0, 11, 29], 50, 'Development50: official states 5-9')]
    for ax, (names, values, n, title) in zip(axes, panels):
        x = np.arange(len(values)); ax.bar(x, np.array(values) / n, color=['#667085', '#2563eb', '#d97706', '#15803d'][:len(values)])
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0, 1.15); ax.set_ylabel('Closed-loop success rate'); ax.set_title(title, fontsize=10)
        for i, value in enumerate(values): ax.text(i, value / n + .025, str(value) + '/' + str(n), ha='center', fontsize=9)
    fig.suptitle('Measured AWQ progress; protected modules BF16; fake quant', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .9))
    for ext in ('png', 'svg'): fig.savefig(ASSETS / ('01_baselines.' + ext), dpi=180)
    plt.close(fig)
    positive = read_json(LATEST / 'data/language_all_attention_complementarity.json')['summary']
    assert [positive[k] for k in ['episodes', 'a_successes', 'b_successes', 'outcome_oracle_successes', 'a_only', 'b_only']] == [50, 20, 29, 34, 5, 14]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(3), [positive[k]/float(positive['episodes']) for k in ['a_successes', 'b_successes', 'outcome_oracle_successes']], color=['#2563eb', '#15803d', '#9ca3af'])
    ax.set_xticks(range(3)); ax.set_xticklabels(['All language no clip', 'Attention no clip', 'Outcome oracle\nnot a deployed router'], fontsize=9)
    for i, value in enumerate([20, 29, 34]): ax.text(i, value/50. + .025, str(value)+'/50', ha='center')
    ax.set_ylim(0, 1); ax.set_ylabel('Closed-loop success rate')
    ax.set_title('Development-only complementarity: 5 and 14 unique successes')
    fig.tight_layout()
    for ext in ('png', 'svg'): fig.savefig(ASSETS / ('02_router_upper_bound.' + ext), dpi=180)
    plt.close(fig)
    datasets = []
    paths = set(ARCHIVE.glob('data/*.csv')) | set(ARCHIVE.glob('data/*.json'))
    for session in (REPO / 'reports/sessions').iterdir():
        if session.is_dir():
            paths.update(session.glob('data/*.csv')); paths.update(session.glob('data/*.json'))
            paths.update(session.glob('data/profile-provenance/*.json'))
    # Retain the actual raw metrics for controls, rescue and initialization probes.
    for root in ['awq-block-audit-20260916', 'awq-visual-diagnostics-20260915', 'awq-baseline-gate-20260916',
                 '107-diagnostics-20260916', '107-followup-20260916', '107-response-svd-20260917']:
        for path in (REPO / 'results' / root).rglob('*.json'):
            if path.name in ['metrics.json', 'summary.json', 'controls.json', 'block-error-summary.json', 'e2e-interventions-old-summary.json']:
                paths.add(path)
    for path in sorted(paths):
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream); records = list(reader); columns = reader.fieldnames
            content = {'columns': columns, 'rows': records, 'row_count': len(records), 'cell_encoding': 'original CSV strings; empty remains empty'}
        else:
            content = read_json(path)
            if relative(path).startswith('results/'):
                content = compact_raw_metrics(content)
        evidence = 'archival_summary' if path.name in ['archival_offline.json', 'historical_sq.csv', 'vision_offline.csv', 'vision_combination_validation.csv', 'e2e-interventions-old-summary.json'] else 'committed_source_snapshot'
        datasets.append({'repository_path': relative(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                         'format': path.suffix[1:], 'evidence_class': evidence, 'data': content})
    images = []
    for text_path in [REPORT / '汇报总结.md', REPORT / '实验日志.md']:
        for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text_path.read_text(encoding='utf-8')):
            path = (REPORT / link).resolve(); assert path.is_file(), 'Missing report image: ' + str(path)
            images.append({'repository_path': relative(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
        for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', text_path.read_text(encoding='utf-8')):
            if link.startswith(('https:', 'http:', '#')): continue
            path = (REPORT / link.split('#')[0]).resolve()
            if path == REPORT / '结果数据.json': continue  # Generated below, then checked in final file inventory.
            assert path.exists(), 'Broken report link: ' + link
    figures = {row['repository_path']: row for row in images}
    for path in ASSETS.glob('*'):
        figures[relative(path)] = {'repository_path': relative(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    data = {'schema_version': '1.0', 'updated_date_beijing': '2026-09-18', 'measurement_cutoff_beijing': '2026-09-18 03:37:12',
        'measurement_source_commit': '3ecaaf4f0f90dff03284e974145976602d345fad',
        'scope': 'Locally available committed VLA-OFT evidence through cutoff; no new GPU experiment in consolidation',
        'protocol': {'model': 'OpenVLA-OFT LIBERO Spatial checkpoint', 'formal_protocol': '10 tasks x 50 official initial states, paired model/env seeds',
                     'implementation': 'BF16 stored fake-quant weights; no packed-kernel efficiency result',
                     'quantized_targets': {'language': 224, 'DINO': 93, 'SigLIP': 105},
                     'teacher_mse': 'relative to BF16 teacher, not ground-truth action error',
                     'repeat_policy': 'Source snapshots may overlap; do not sum datasets as independent observations'},
        'authoritative_baseline': baseline, 'authoritative_scope': scope, 'attention_development_extension_splits': splits,
        'clip_factorial': factorial,
        'contextual_routing': {'positive_development_pair': read_json(LATEST / 'data/language_all_attention_complementarity.json'),
                              'negative_development_pair': read_json(LATEST / 'data/language_g128_g64_complementarity.json'),
                              'trained_router': False, 'oracle_is_deployable': False},
        'not_measured': ['certified current SQ W4A4 closed-loop', 'gradient-trained PEFT closed-loop', 'trained contextual routing',
                         'packed INT2/INT4 total bytes, peak VRAM and synchronized latency', 'complete four-suite reproduction'],
        'sources': datasets, 'figures': sorted(figures.values(), key=lambda r: r['repository_path']),
        'diagnostic_index_policy': 'In raw metrics only, large features/attention/activation_local_error blocks and numeric arrays longer than64 are represented by byte count/length and canonical JSON SHA. Original files remain intact. Action error scalars, per-frame/per-dimension results, CSV tables and report JSON snapshots retain their values.',
        'document_hashes': {name: sha(REPORT / name) for name in ['汇报总结.md', '实验日志.md']},
        'reproduction': 'python scripts/build_weekend_review.py (numpy/matplotlib; no network or GPU)',
        'large_artifacts': 'Source videos/profile/NPZ are in separately SHA-verified server and local archives; models/calibration/original profiles are not fully offsite-backed-up'}
    output = REPORT / '结果数据.json'
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    assert sorted(p.name for p in REPORT.iterdir()) == ['实验日志.md', '汇报总结.md', '结果数据.json']
    print(json.dumps({'files': 3, 'datasets': len(datasets), 'images': len(figures), 'paired_scope_records': len(pairs),
                      'output_bytes': output.stat().st_size, 'validated_main_counts': True}, ensure_ascii=False))

if __name__ == '__main__': build()
