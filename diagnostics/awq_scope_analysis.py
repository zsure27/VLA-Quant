"""Pair completed scope ablations with original BF16 manifests, without reruns."""
import argparse
import json
import re
from pathlib import Path
from baseline_shard_analysis import parse_episodes

def analyze(directory, reference_root):
    directory = Path(directory)
    if (directory / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError('scope batch unfinished or failed')
    cases = [p for p in directory.iterdir() if p.is_dir() and (p/'command.txt').is_file()]
    if len(cases) != 1:
        raise ValueError('one scope case required')
    case = cases[0]
    if (case/'exit-code.txt').read_text().strip() != '0':
        raise ValueError('scope case failed')
    command = (case/'command.txt').read_text()
    offset = int(re.search(r'--initial-state-offset\s+(\d+)', command).group(1))
    count = int(re.search(r'--num_trials_per_task\s+(\d+)', command).group(1))
    logs = list(case.glob('EVAL-*.txt'))
    if len(logs) != 1:
        raise ValueError('one original scope EVAL log required')
    rows = parse_episodes(logs[0].read_text())
    expected = {(t,i) for t in range(10) for i in range(offset,offset+count)}
    if {(r['task_id'],r['init_state_index']) for r in rows} != expected:
        raise ValueError('wrong or incomplete scope coverage')
    contract = (directory/'CONTRACT_SHA256SUMS.txt').read_text().splitlines()[:2]
    reference = {}
    for shard in Path(reference_root).glob('awq-baseline-shard-*'):
        if not (shard/'paired-results.json').is_file():
            continue
        if (shard/'CONTRACT_SHA256SUMS.txt').read_text().splitlines()[:2] != contract:
            raise ValueError('reference evaluator contract differs')
        for log in (shard/'bf16').glob('EVAL-*.txt'):
            for r in parse_episodes(log.read_text()):
                key = (r['task_id'],r['init_state_index'])
                if key not in expected:
                    continue
                if key in reference:
                    raise ValueError('duplicate reference initial state')
                reference[key] = (r,str(shard))
    if set(reference) != expected:
        raise ValueError('reference coverage missing')
    paired = []
    for r in rows:
        b,source = reference[(r['task_id'],r['init_state_index'])]
        manifest = lambda x: {k:v for k,v in x.items() if k not in ('success','episode_errors')}
        if manifest(b) != manifest(r):
            raise ValueError('reference initial-state/seed manifest differs')
        paired.append(dict(task_id=r['task_id'],init_state_index=r['init_state_index'],
            init_state_sha256=r['init_state_sha256'],model_seed=r['model_seed'],env_seed=r['env_seed'],
            bf16_success=b['success'],candidate_success=r['success'],
            candidate_episode_errors=r['episode_errors'],reference_episode_errors=b['episode_errors'],
            reference_shard=source))
    return dict(directory=str(directory),case=case.name,paired_episodes=paired,
        summary=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
            episode_errors=sum(len(r['episode_errors']) for r in rows),
            bf16_successes=sum(r['bf16_success'] for r in paired)),
        note='Scope ablation; protected heads remain BF16; not uniform whole-model W2 or PEFT training.')

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--reference-root',type=Path,required=True);a=p.parse_args()
    r=analyze(a.directory,a.reference_root)
    (a.directory/'scope-paired-results.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['summary']))
