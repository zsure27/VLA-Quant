import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"diagnostics"))
from baseline_shard_analysis import parse_episodes
class ShardAnalysisTest(unittest.TestCase):
 def fixture(self,index=5):return "EPISODE_MANIFEST "+json.dumps(dict(task_id=0,init_state_index=index))+"\nSuccess: True\n"
 def test_actual_index_and_pending_not_counted(self):
  rows=parse_episodes(self.fixture()+"EPISODE_MANIFEST "+json.dumps(dict(task_id=0,init_state_index=6)))
  self.assertEqual(len(rows),1);self.assertEqual(rows[0]["init_state_index"],5)
 def test_repeated_official_index_rejected(self):
  with self.assertRaises(ValueError):parse_episodes(self.fixture()*2)
 def test_error_remains_in_denominator(self):
  text=self.fixture().replace("Success: True","Episode error: test error\nSuccess: False")
  rows=parse_episodes(text);self.assertEqual(len(rows),1);self.assertFalse(rows[0]["success"]);self.assertEqual(len(rows[0]["episode_errors"]),1)
 def test_unpaired_outcome_rejected(self):
  with self.assertRaises(ValueError):parse_episodes("Success: True")
if __name__=="__main__":unittest.main()
