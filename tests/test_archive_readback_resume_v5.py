import hashlib,importlib.util,io,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('archive_v5_test',pathlib.Path(__file__).resolve().parents[1]/'scripts/github_archive_store_v5.py');v5=importlib.util.module_from_spec(spec);spec.loader.exec_module(v5)
class Response(io.BytesIO):
 def __init__(self,b,status=200,range_header=None):
  super().__init__(b);self.status=status;self.headers={'Content-Type':'application/octet-stream'}
  if range_header:self.headers['Content-Range']=range_header
 def geturl(self):return 'https://release-assets.githubusercontent.com/test/asset?signature=opaque'
class Resume(unittest.TestCase):
 def run_case(self,folder,response):
  data=b'abcdefgh';digest=hashlib.sha256(data).hexdigest();part=types.SimpleNamespace(name='part',size=8,sha256=digest,index=0)
  client=object.__new__(v5.ResumableReadback);client.cache=pathlib.Path(folder);(client.cache/'part.readback').write_bytes(b'abc');client.open_download=lambda _:Response(data)
  def request(req,**kwargs):
   self.assertEqual(req.get_header('Range'),'bytes=3-');self.assertFalse(req.has_header('Authorization'));return response
  with patch.object(v5.urllib.request,'urlopen',side_effect=request):client.verify_download({},part)
 def test_exact_range_resume_and_full_hash(self):
  with tempfile.TemporaryDirectory() as folder:
   self.run_case(folder,Response(b'defgh',206,'bytes 3-7/8'));self.assertEqual((pathlib.Path(folder)/'part.readback').read_bytes(),b'abcdefgh')
 def test_ignored_range_rejected(self):
  with tempfile.TemporaryDirectory() as folder:
   with self.assertRaises(v5.core.ArchiveError):self.run_case(folder,Response(b'abcdefgh',200))
   self.assertEqual((pathlib.Path(folder)/'part.readback').read_bytes(),b'abc')
 def test_corrupt_equal_length_rejected(self):
  with tempfile.TemporaryDirectory() as folder:
   with self.assertRaises(v5.core.HashMismatch):self.run_case(folder,Response(b'defgZ',206,'bytes 3-7/8'))
if __name__=='__main__':unittest.main()
