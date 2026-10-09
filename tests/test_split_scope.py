"""Reject data leakage and claim drift; ensure the cheaper plan keeps fresh pairing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qvla.split_scope import ROOT, audit_scope, task_inventory, sha


class SplitScopeTests(unittest.TestCase):
    def setUp(self):
        self.train=[{'task_id':0,'init_state_index':0,'init_state_sha256':'train-init','observation_sha256':'train-obs'}]
        self.evaluation=[{'task_id':0,'init_state_index':20,'init_state_sha256':'eval-init',
                          'observation_sha256_all_queries':['eval-obs']}]

    def test_same_task_is_valid_development_but_not_task_transfer(self):
        self.assertEqual(audit_scope(self.train,self.evaluation)['shared_tasks'],[0])
        with self.assertRaisesRegex(ValueError,'trained tasks'):
            audit_scope(self.train,self.evaluation,'recovery_task_transfer_diagnostic',[],[])
        with self.assertRaises(ValueError):audit_scope(self.train,self.evaluation,'final_evaluation')

    def test_different_reset_or_filename_cannot_hide_content_overlap(self):
        for field,value,pattern in [('init_state_sha256','train-init','initial-state content'),
                                    ('observation_sha256_all_queries',['train-obs'],'observation content'),
                                    ('observation_sha256_all_queries',[],'Missing observations')]:
            rows=copy.deepcopy(self.evaluation);rows[0][field]=value
            with self.assertRaisesRegex(ValueError,pattern):audit_scope(self.train,rows)

    def test_heldout_task_cannot_enter_svd_or_initialization(self):
        rows=copy.deepcopy(self.evaluation);rows[0]['task_id']=1
        for calib,init in (([1],[]),([],[1])):
            with self.assertRaisesRegex(ValueError,'exposed'):
                audit_scope(self.train,rows,'recovery_task_transfer_diagnostic',calib,init)
        self.assertEqual(audit_scope(self.train,rows,'recovery_task_transfer_diagnostic',[0],[0])['shared_tasks'],[])

    def test_actual_b3_training_initials_disjoint_from_registered_development(self):
        rows=json.loads((ROOT/'results/experiments/p2-shared-peft/20261008-046-b2-b3/B3-student-state80-manifest.json').read_text())['samples']
        inventory=task_inventory()
        self.assertEqual(len({(r['task_id'],r['init_state_index']) for r in rows}),40)
        self.assertTrue(all(r['init_state_sha256']==inventory[r['task_id']]['active_state_sha256'][r['init_state_index']] for r in rows))
        self.assertFalse({r['init_state_sha256'] for r in rows}&{h for t in inventory.values() for h in t['active_state_sha256'][20:40]})

    def test_changed_data_receipt_source_rejected(self):
        from scripts.audit_b4_data_scope import verify_receipt
        (ROOT/'backups').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'backups') as tmp:
            p=Path(tmp);source=p/'source.json';source.write_text('{}')
            receipt=p/'receipt.json';receipt.write_text(json.dumps({'gate':'PASS_DATA_SCOPE',
                'claim':'within_task_development','source_sha256':{str(source):sha(source)}}))
            verify_receipt(receipt);source.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'source changed'):verify_receipt(receipt)

    def test_v2_only_fresh_pair_400_episodes_and_cpu_admission_first(self):
        from scripts.prepare_b4_pair_v2_plans import build_plans
        from scripts.vla_stage_supervisor import validate_plan
        (ROOT/'backups').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'backups') as tmp:
            d=Path(tmp);out=d/'plans';out.mkdir()
            m={k:str(d/k) for k in ('checkpoint','trajectory_split','w2_g128','w2_g64','w4',
                'prior_closure_receipt','b3_train_dir','student80','calibration80','oft_root','libero_root','official_root')}
            with patch('scripts.prepare_b4_joint_plans.sha',return_value='fixture'):
                plans=build_plans(m,d/'session',out,'python','verified-host')
            total={'B3':0,'B4':0}
            for phase,plan in plans.items():
                validate_plan(plan)
                for stage in plan['stages']:
                    cmd=stage['command']
                    if '--local_log_dir' in cmd:
                        case=cmd[cmd.index('--model-id')+1]
                        self.assertIn(case,total)
                        if not stage['id'].startswith('micro-'):total[case]+=10*int(cmd[cmd.index('--num_trials_per_task')+1])
                    if stage['id'].startswith('audit-') or stage['id'].startswith('verify-first') or stage['id'].startswith('verify-next'):
                        self.assertIn(str(ROOT/'scripts/audit_b4_pair_v2.py'),cmd)
            self.assertEqual(total,{'B3':200,'B4':200})
            self.assertEqual(plans['first50']['stages'][0]['id'],'verify-data-scope')

    def test_v2_negative_result_passes_protocol_but_observation_leakage_fails(self):
        from scripts.audit_b4_pair_v2 import audit
        (ROOT/'backups').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'backups') as tmp:
            root=Path(tmp); training=root/'training.json'
            training.write_text(json.dumps({'samples':self.train}))
            keys=[(t,20,str(t),1,t,'paired') for t in range(10)]
            reference=[{'task_id':t,'init_state_index':20,'key':list(keys[t]),'first':str(t),
                        'A4':False,'BF16':True,'A0':True} for t in range(10)]
            receipt={'training_manifest':str(training),'reference_rows':reference}
            def result(path,case,offset,count):
                return {'keys':keys,'first':{i:str(i) for i in range(10)},
                        'success':[case=='B3']*10,'sha256':{}}
            for case in ('B3','B4'):
                d=root/'eval-micro'/case;d.mkdir(parents=True)
                (d/'on-policy-events.jsonl').write_text(json.dumps({'record_type':'query','observation_sha256':'eval-obs'})+'\n')
            with patch('scripts.audit_b4_pair_v2.verify_receipt',return_value=receipt),patch('scripts.audit_b4_pair_v2.audit_case',side_effect=result):
                (root/'data-scope-receipt.json').write_text('{}')
                record=audit(root,'micro')
                self.assertEqual(record['primary']['break'],10)
                self.assertEqual(record['gate'],'PASS_PROTOCOL_B4_PAIR_V2_MICRO')
                (root/'eval-micro/B4/on-policy-events.jsonl').write_text(json.dumps({'record_type':'query','observation_sha256':'train-obs'})+'\n')
                with self.assertRaisesRegex(ValueError,'training observation'):audit(root,'micro')


if __name__=='__main__':unittest.main()
