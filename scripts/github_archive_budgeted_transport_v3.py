"""Budgeted serial payload transfer: bounded GET retries, single POST, no cached SHA claims."""
import json,os,time,threading
from pathlib import Path
from scripts.github_archive_curl_transport_v2 import GitHubCurlV2
from scripts import github_archive_store_v2 as core

class BudgetedGitHubCurlV3(GitHubCurlV2):
    def __init__(self,token,cache,curl,*,traffic_ledger,budget_bytes,base_reserved_bytes,initial_extra_bytes):
        self.ledger_path=Path(traffic_ledger);self.payload_lock=threading.RLock()
        self.traffic=json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else dict(budget_bytes=budget_bytes,base_reserved_bytes=base_reserved_bytes,extra_failure_bytes_upper_bound=initial_extra_bytes,failures=[])
        assert self.traffic['budget_bytes']==budget_bytes and self.traffic['base_reserved_bytes']==base_reserved_bytes
        assert self.traffic['extra_failure_bytes_upper_bound']>=initial_extra_bytes
        self._save_traffic()
        super().__init__(token,cache,curl)

    def _save_traffic(self):
        tmp=self.ledger_path.with_suffix('.tmp');tmp.write_text(json.dumps(self.traffic,indent=2));os.replace(tmp,self.ledger_path)

    def _room(self,part):
        upper=self.traffic['base_reserved_bytes']+self.traffic['extra_failure_bytes_upper_bound']
        if upper+part.size>self.traffic['budget_bytes']:
            raise core.ArchiveError('No remaining traffic insurance for another payload attempt')

    def _failed(self,part,operation,error):
        self.traffic['extra_failure_bytes_upper_bound']+=part.size
        assert self.traffic['base_reserved_bytes']+self.traffic['extra_failure_bytes_upper_bound']<=self.traffic['budget_bytes']
        self.traffic['failures'].append(dict(operation=operation,part=part.name,maximum_bytes=part.size,error_type=type(error).__name__,observed_at=time.time()))
        self._save_traffic()

    def upload(self,part):
        with self.payload_lock:
            self._room(part)
            try:return super().upload(part)
            except Exception as e:
                self._failed(part,'POST',e)
                raise

    def verify_download(self,asset,part):
        with self.payload_lock:
            for attempt in range(3):
                self._room(part)
                try:return super().verify_download(asset,part)
                except Exception as e:
                    self._failed(part,'GET_READBACK',e)
                    if attempt==2 or not core.retryable(e):raise
                    time.sleep(3*(attempt+1))
