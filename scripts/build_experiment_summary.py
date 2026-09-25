"""Assemble three-file review from committed measurements; no server/network."""
import csv
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / 'reports/summaries/20260918-experiment-overview'
ARCHIVE = REPO / 'reports/archive/20260916-weekend-review'
ASSETS = REPORT / 'assets'
LATEST = REPO / 'reports/experiments/p0-foundation-baselines/20260918-107-awq-continuation'
BASELINE = REPO / 'reports/experiments/p0-foundation-baselines/20260917-107-baseline-continuation'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def relative(path):
    return path.relative_to(REPO).as_posix()

def sha(path):
    content = path.read_bytes()
    if path.suffix.lower() in ['.csv', '.json', '.md', '.py', '.txt', '.svg']:
        content = content.replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()

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
    plt.rcParams['svg.hashsalt'] = 'vla-weekend-review-20260918'
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
    for ext in ('png', 'svg'):
        fig.savefig(ASSETS / ('01_baselines.' + ext), dpi=180, **({'metadata': {'Date': None}} if ext == 'svg' else {}))
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
    for ext in ('png', 'svg'):
        fig.savefig(ASSETS / ('02_router_upper_bound.' + ext), dpi=180, **({'metadata': {'Date': None}} if ext == 'svg' else {}))
    plt.close(fig)
    datasets = []
    paths = set(ARCHIVE.glob('data/*.csv')) | set(ARCHIVE.glob('data/*.json'))
    for session in (REPO / 'reports/experiments').glob('*/*'):
        if session.is_dir():
            paths.update(session.glob('data/*.csv')); paths.update(session.glob('data/*.json'))
            paths.update(session.glob('data/profile-provenance/*.json'))
    # Retain the actual raw metrics for controls, rescue and initialization probes.
    raw_roots = [
        'results/experiments/p0-foundation-baselines/20260916-awq-block-audit',
        'results/experiments/p0-foundation-baselines/20260915-awq-visual-diagnostics',
        'results/experiments/p0-foundation-baselines/20260916-awq-baseline-gate',
        'results/experiments/p0-foundation-baselines/20260916-107-diagnostics',
        'results/experiments/p0-foundation-baselines/20260916-107-followup',
        'results/experiments/p2-shared-peft/20260917-107-response-svd',
    ]
    for root in raw_roots:
        for path in (REPO / root).rglob('*.json'):
            if path.name in ['metrics.json', 'summary.json', 'controls.json', 'block-error-summary.json', 'e2e-interventions-old-summary.json']:
                paths.add(path)
    for path in sorted(paths):
        entry = {'repository_path': relative(path), 'sha256': sha(path),
                 'bytes': path.stat().st_size, 'format': path.suffix[1:]}
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                entry['columns'] = reader.fieldnames
                entry['row_count'] = sum(1 for _ in reader)
        entry['evidence_class'] = ('archival_summary' if path.name in [
            'archival_offline.json', 'historical_sq.csv', 'vision_offline.csv',
            'vision_combination_validation.csv', 'e2e-interventions-old-summary.json']
            else 'committed_source_snapshot')
        datasets.append(entry)
    images = []
    for text_path in [REPORT / 'README_CN.md', REPORT / 'EXPERIMENT_LOG_CN.md']:
        for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text_path.read_text(encoding='utf-8')):
            path = (REPORT / link).resolve(); assert path.is_file(), 'Missing report image: ' + str(path)
            images.append({'repository_path': relative(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
        for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', text_path.read_text(encoding='utf-8')):
            if link.startswith(('https:', 'http:', '#')): continue
            path = (REPORT / link.split('#')[0]).resolve()
            if path == REPORT / 'EVIDENCE_INDEX.json': continue  # Generated below, then checked in final file inventory.
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
        'file_hash_policy': 'SHA256 of canonical LF bytes for CSV/JSON/Markdown/Python/text/SVG; binary PNG uses original bytes. bytes fields record local source size. Embedded profile and upstream hashes retain their own original conventions.',
        'diagnostic_index_policy': 'Sources are indexed by relative path, SHA256 and byte count. Large arrays and CSV rows remain in their source files and are not duplicated here. Authoritative result summaries above retain the measured counts.',
        'document_hashes': {name: sha(REPORT / name) for name in ['README_CN.md', 'EXPERIMENT_LOG_CN.md']},
        'reproduction': 'python scripts/build_experiment_summary.py (numpy/matplotlib; no network or GPU)',
        'large_artifacts': 'Source videos/profile/NPZ are in separately SHA-verified server and local archives; models/calibration/original profiles are not fully offsite-backed-up'}
    output = REPORT / 'EVIDENCE_INDEX.json'
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    assert sorted(p.name for p in REPORT.iterdir()) == ['EVIDENCE_INDEX.json', 'EXPERIMENT_LOG_CN.md', 'README_CN.md', 'assets']
    print(json.dumps({'summary_files': 3, 'asset_directory': relative(ASSETS), 'datasets': len(datasets), 'images': len(figures), 'paired_scope_records': len(pairs),
                      'output_bytes': output.stat().st_size, 'validated_main_counts': True}, ensure_ascii=False))

if __name__ == '__main__': build()
