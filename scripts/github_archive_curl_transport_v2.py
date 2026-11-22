"""Single-POST private archive transport; no Expect handshake or hidden upload retries."""
from scripts.github_archive_curl_transport_v1 import GitHubCurl
from scripts import github_archive_store_v2 as core

class GitHubCurlV2(GitHubCurl):
    def request(self,url,headers,*,method='GET',payload=None):
        supplied=dict(headers)
        if method=='POST' and payload is not None:
            supplied['Expect']=''  # Stream without the proxy-sensitive 100-continue handshake.
        return super().request(url,supplied,method=method,payload=payload)

    def ensure_asset(self,part):
        asset=self.get_asset(part.name)
        if asset is not None:
            if asset.get('state')!='uploaded':
                raise core.ArchiveError('Existing unverified starter retained; no retransmission')
            core.validate_asset(asset,part)
            return asset
        try:
            # The API enforces unique asset names. A race/lost response is resolved
            # with one read-only listing; never repeat a payload blindly.
            asset=self.upload(part)
            core.validate_asset(asset,part)
            return asset
        except Exception:
            self.refresh_assets()
            asset=self.get_asset(part.name)
            if asset is not None and asset.get('state')=='uploaded':
                core.validate_asset(asset,part)
                return asset
            raise
