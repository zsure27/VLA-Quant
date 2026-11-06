"""CPU-only sampling/sharding specification for independent trajectory diagnosis.

No model import, server connection, GPU launcher, DyQ algorithm, or OPD training.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from qvla.runtime_error_protocol_v4 import complete_case_ids

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
MODELS=('pi05','groot_n15','openvla_oft')
CONDITIONS=('BF16','GPTQ-W4A16','AWQ-W4A16','OmniQuant-W4A16','ActQuant-W4A16',
            'SQ-W4A8','DuQuant-W4A8','QuantVLA-W4A8','SQ-W4A4','DuQuant-W4A4',
            'QuantVLA-W4A4','HoloQ-VLA-W4A4')


def validate(study):
    if study.get('executable') is not False:raise ValueError('A sampling specification must remain non-executable')
    if study['conditions']!=list(CONDITIONS):raise ValueError('Versioned static method catalogue required; no QVLA/DyQ aliases')
    if [x['id'] for x in study['models']]!=list(MODELS):raise ValueError('Expected three registered model identities')
    if study['suites']!=list(SUITES) or study['task_ids']!=list(range(10)):raise ValueError('Four suites and all ten tasks required')
    indices=study['reset_indices']
    if len(indices)!=5 or len(set(indices))!=5 or any(type(i) is not int or i<0 for i in indices):
        raise ValueError('Five distinct nonnegative initial-state indices required')
    discovery=study['analysis']['discovery_reset_indices'];confirmation=study['analysis']['confirmation_reset_indices']
    if len(discovery)!=2 or len(confirmation)!=3 or set(discovery)&set(confirmation) or set(discovery+confirmation)!=set(indices):
        raise ValueError('Two exploration and three locked confirmation initials required')
    if study['training'] or study['opd'] or study['dynamic_routing']:raise ValueError('Observation-only study required')
    if study['analysis']['reference']!='matched_reset_quantized_vs_bf16_natural_trajectory_difference' or study['perturbation']['enabled_by_default']:
        raise ValueError('Natural pre/post-quantization diagnosis required; no automatic DyQ perturbation replication')
    if study['analysis']['complete_cohort_shared_across_time_and_precision']:
        raise ValueError('Do not globally intersect every quantizer cohort')
    return study


def planned_cases(study):
    validate(study)
    rows=[]
    for initial in study['reset_indices']:
        for suite in study['suites']:
            for task in study['task_ids']:
                rows.append(dict(case_id=f'{suite}-task{task}-reset{initial}',suite=suite,task_id=task,
                    reset_index=initial,phase='discovery' if initial in study['analysis']['discovery_reset_indices'] else 'confirmation',
                    role='development',initial_state_sha256=None,task_name=None,
                    status='PLANNED_NOT_RUN_INITIAL_STATE_CONTENT_PENDING'))
    return rows


def budget(study):
    cases=planned_cases(study);n=len(cases);models=len(study['models']);conditions=len(study['conditions'])
    return dict(cases_per_model_condition=n,cases_per_suite_model_condition=n//4,
                model_condition_cells=models*conditions,natural_episode_upper_bound=n*models*conditions,
                bf16_reference_episodes=models*n,quantized_episode_upper_bound=models*n*(conditions-1),
                bf16_reference_reused_in_analysis_not_rerun_per_method=True,
                all_conditions_admitted=False,short_probe_episodes_registered=0,
                automatic_gpu_dispatch=False)


def gpu_shards(study,slots):
    """Model x condition x suite x initial; each shard covers all ten tasks.

    Assignment is a deterministic CPU scheduling proposal, not authorization.
    BF16 appears once, not once for every quantizer reference comparison.
    """
    validate(study)
    if type(slots) is not int or slots<1:raise ValueError('Positive integer GPU-slot count required')
    result=[]
    for model in MODELS:
        for condition in CONDITIONS:
            for initial in study['reset_indices']:
                for suite in SUITES:
                    key=f'{model}/{condition}/{suite}/reset{initial}'
                    result.append(dict(shard_id=hashlib.sha256(key.encode()).hexdigest()[:20],
                        model=model,condition=condition,suite=suite,reset_index=initial,task_ids=list(range(10)),
                        planned_episodes=10,proposed_gpu_slot=len(result)%slots,
                        output_relative=key,status='BLOCKED_NATIVE_ADMISSION_AND_ROUND_AUTHORIZATION_REQUIRED'))
    return result


def pairwise_cohort(rows,model,direction,condition,times,expected_case_ids=None):
    """Do not let an unrelated failed method erase every method's cohort.

    This helper only selects records; an independent raw-data admission must
    validate identities, time alignment, finiteness and outcomes before use.
    """
    if model not in MODELS or condition not in CONDITIONS or condition=='BF16':raise ValueError('A registered static quantizer and model are required')
    selected=[r for r in rows if r['model']==model and r['direction']==direction and r['precision'] in ('BF16',condition)]
    complete,risk=complete_case_ids(selected,['BF16',condition],times)
    observed={r['case_id'] for r in selected}
    planned=set(expected_case_ids) if expected_case_ids is not None else observed
    if not observed<=planned:raise ValueError('Records outside the preregistered case set')
    return dict(records=[r for r in selected if r['case_id'] in complete],complete_case_ids=sorted(complete),
                complete_n=len(complete),planned_n=len(planned),never_observed_case_ids=sorted(planned-observed),
                risk_set=risk,interpretation='Conditional complete-pair diagnostic; not a shared-cohort ranking across methods')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path(__file__).resolve().parents[1]/'configs/infra/runtime_commonality_v5/study.json')
    p.add_argument('--gpu-slots',type=int,default=1,help='CPU assignment proposal only; actual GPU UUIDs remain unbound')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory')
    study=validate(json.loads(args.study.read_text(encoding='utf-8')))
    cases=planned_cases(study);shards=gpu_shards(study,args.gpu_slots);summary=budget(study)
    args.output.mkdir(parents=True)
    with (args.output/'planned-cases.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]),lineterminator='\n');writer.writeheader();writer.writerows(cases)
    receipt=dict(status='CPU_SAMPLING_ONLY_NO_SERVER',study_sha256=hashlib.sha256(args.study.read_bytes()).hexdigest(),
                 summary=summary,gpu_slot_proposal=args.gpu_slots,actual_gpu_uuids=None,measured_episodes=0)
    (args.output/'sampling-summary.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
    (args.output/'planned-shards.json').write_bytes((json.dumps(shards,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(summary))


if __name__=='__main__':main()
