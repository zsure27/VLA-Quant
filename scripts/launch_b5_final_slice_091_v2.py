"""Launch registered final slice after completed protocol and paired analysis."""
import datetime, hashlib, json, shutil, socket, subprocess
from pathlib import Path

ROOT = Path('/root/autodl-tmp/qvla-repro')
BASE = ROOT/'backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
V3 = ROOT/'control/b5-pilot-091-20261012-continuation-v3'
SOURCE = ROOT/'control/b5-pilot-091-20261012-v2/source'
PYTHON = '/root/miniconda3/envs/qvla-oft/bin/python'
assert socket.gethostname() == 'autodl-container-dfbc4bb0ff-1e85dd94'
prior = V3/'plans/control/20261012-091-b5-pilot-v2-reset-30-39-continuation-v3/status.json'
state = json.loads(prior.read_text())
assert state['state'] == 'AWAITING_GATE_REVIEW' and state['last_exit_code'] == 0
assert len(state['completed_stage_ids']) == 8
gate = json.loads((BASE/'reset-30-39-gate.json').read_text())
assert gate['gate'] == 'PASS_B5_PHASE_PROTOCOL' and (gate['offset'], gate['count']) == (30,10)
analysis = json.loads((BASE/'combined350-paired-analysis.json').read_text())
assert analysis['gate'] == 'PASS_AUDITED_DISJOINT_B5_PAIRED_COMBINED' and analysis['episodes'] == 350
assert len({(r['task_id'],r['init_state_index']) for r in analysis['rows']}) == 350
plan = V3/'plans/reset-40-49-plan.json'
file_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
assert file_sha == 'ecb7b89d4a4b35e2ab2fa91ae480940abcda6ddd06b323c8b6f3c90ecf385990'
registered = json.loads(plan.read_text())
status = Path(registered['control_root'])/registered['plan_id']/'status.json'
assert not status.exists(), 'Already started; do not repeat GPU'
# Executing source is a Git archive, so verify deployed provenance and pinned files.
deployment = json.loads((V3/'deployment.json').read_text())
assert deployment['gate'] == 'PASS_REMOTE_CONTINUATION_PLAN_SHA_AND_OUTPUT_PATHS'
assert deployment['source_commit'] == 'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d'
lock = json.loads((ROOT/'control/b5-pilot-091-20261012-v2/plans/runtime-lock.json').read_text())
assert lock['expected_hostname'] == socket.gethostname() and lock['files']
for name, expected in lock['files'].items():
    digest = hashlib.sha256()
    with Path(name).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): digest.update(chunk)
    assert digest.hexdigest() == expected, 'Locked input/code changed: '+name
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
free = shutil.disk_usage(ROOT).free
assert free >= 12*1024**3
receipt = BASE/'reset-40-49-v3-launch-receipt.json'
assert not receipt.exists()
log = V3/'reset-40-49-runner.log'
with log.open('x') as stream:
    process = subprocess.Popen([PYTHON,str(SOURCE/'scripts/vla_stage_supervisor.py'),str(plan)],cwd=SOURCE,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value = {'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runner_pid':process.pid,'plan':str(plan),'status':str(status),'log':str(log),'plan_file_sha256':file_sha,'free_bytes':free,'previous_gate':gate['gate'],'continuation':'Preregistered protocol/resource/quota, not success direction'}
with receipt.open('x') as stream: json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
