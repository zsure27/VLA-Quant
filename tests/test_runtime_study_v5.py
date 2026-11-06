import copy
import json
from pathlib import Path
import unittest
from qvla.runtime_study_v5 import planned_cases,budget,gpu_shards,pairwise_cohort,validate


class RuntimeStudyV5(unittest.TestCase):
    def study(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/infra/runtime_commonality_v5/study.json').read_text(encoding='utf-8'))

    def test_sampling_covers_40_tasks_five_distinct_initials_without_training(self):
        study=self.study();cases=planned_cases(study)
        self.assertEqual(len(cases),200);self.assertEqual(len({x['case_id'] for x in cases}),200)
        for suite in study['suites']:
            for task in study['task_ids']:
                self.assertEqual({x['reset_index'] for x in cases if x['suite']==suite and x['task_id']==task},set(study['reset_indices']))
        bad=copy.deepcopy(study);bad['reset_indices'][-1]=bad['reset_indices'][0]
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(study);bad['perturbation']['enabled_by_default']=True
        with self.assertRaises(ValueError):validate(bad)

    def test_gpu_sharding_never_reruns_shared_fp_reference_or_authorizes_gpu(self):
        study=self.study();shards=gpu_shards(study,4)
        self.assertEqual(sum(x['planned_episodes'] for x in shards),7200)
        self.assertEqual(sum(x['planned_episodes'] for x in shards if x['condition']=='BF16'),600)
        self.assertEqual(len({x['output_relative'] for x in shards}),len(shards))
        self.assertTrue(all(x['status'].startswith('BLOCKED_') and x['proposed_gpu_slot'] in range(4) for x in shards))
        self.assertFalse(budget(study)['automatic_gpu_dispatch'])

    def test_unrelated_failed_method_does_not_erase_pair_and_missing_case_stays_counted(self):
        rows=[dict(model='openvla_oft',direction='Tz+',precision=p,case_id=c,tau=t)
              for p in ('BF16','SQ-W4A8') for c in ('a','b') for t in (0,1)]
        rows.append(dict(model='openvla_oft',direction='Tz+',precision='AWQ-W4A16',case_id='a',tau=0))
        result=pairwise_cohort(rows,'openvla_oft','Tz+','SQ-W4A8',[0,1],['a','b','c'])
        self.assertEqual(result['complete_n'],2);self.assertEqual(result['planned_n'],3)
        self.assertEqual(result['never_observed_case_ids'],['c'])
        with self.assertRaises(ValueError):pairwise_cohort(rows,'openvla_oft','Tz+','DyQ-VLA-W4-Adyn',[0,1])


if __name__=='__main__':unittest.main()
