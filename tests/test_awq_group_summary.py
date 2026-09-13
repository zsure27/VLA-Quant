"""用合成结果验证分组汇总可执行且拒绝语言误改。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('group_summary', Path(__file__).resolve().parents[1]/'scripts/summarize_primary_group.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTest(unittest.TestCase):
    def test_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            language = [f'language_model.{i}' for i in range(224)]
            primary = [f'vision_backbone.featurizer.{i}' for i in range(93)]
            for name, group in [('primary-g128',128),('primary-g64',64)]:
                p=root/name
                p.mkdir(parents=True)
                rows=[dict(sample=str(i),normalized_action={'mse':0.1},raw_gripper_disagreement=0) for i in range(32)]
                scope=dict(weight_targets=language+primary,activation_targets=[],weight_bits_by_target={n:2 for n in language+primary},
                    group_size_by_target={**{n:128 for n in language},**{n:group for n in primary}},
                    removed_clip_targets=language[:160],removed_visual_clip_targets=[])
                (p/'metrics.json').write_text(json.dumps(rows))
                (p/'scope.json').write_text(json.dumps(scope))
            with patch.object(sys,'argv',['summary',str(root)]):
                module.main()
            self.assertTrue(Path(str(root)+'-review.zip').exists())
            scope['group_size_by_target'][language[0]]=64
            (root/'primary-g64'/'scope.json').write_text(json.dumps(scope))
            with patch.object(sys,'argv',['summary',str(root)]), self.assertRaises(ValueError):
                module.main()
