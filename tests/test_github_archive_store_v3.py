import hashlib,json,sys,tempfile,unittest,threading
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import github_archive_store_v3 as v3
from test_github_archive_store_v2 import FakeGitHub,RELEASE
class SmallPartTransportTests(unittest.TestCase):
 def test_network_failure_prevents_later_queued_uploads(self):
  client=v3.FailFastGitHub.__new__(v3.FailFastGitHub)
  client.failed=threading.Event()
  with patch.object(v3.core.GitHub,'ensure_asset',side_effect=ConnectionError('offline')) as request:
   with self.assertRaises(ConnectionError):client.ensure_asset(None)
   with self.assertRaises(v3.core.ArchiveError):client.ensure_asset(None)
   self.assertEqual(request.call_count,1)
 def test_layout_isolation_and_worker64_roundtrip_resume(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'raw.tar';p.write_bytes(bytes(range(200)))
   plan={'repository':v3.core.REPO,'security_scan_passed':True,'files':[{'path':str(p),'bytes':200,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]}
   small=v3.core.prepare_parts(plan,chunk_size=4);old=v3.core.prepare_parts(plan,chunk_size=8)
   bad={'files':{old[0].whole_sha256:{'parts':[{'name':old[0].name}]}}}
   with self.assertRaises(v3.core.ArchiveError):v3.validate_receipt_layout(bad,small)
   client=FakeGitHub();receipt=v3.core.Receipt(root/'receipt.json',RELEASE)
   v3.core.run_parts(small,client,receipt,workers=64)
   self.assertTrue(receipt.state['files'][small[0].whole_sha256]['remote_download_sha256_verified'])
   before=len(client.downloads);v3.core.run_parts(small,client,receipt,workers=64)
   self.assertEqual(before,len(client.downloads))
   with self.assertRaises(v3.core.ArchiveError):v3.core.run_parts(small,client,receipt,workers=65)
if __name__=='__main__':unittest.main()
