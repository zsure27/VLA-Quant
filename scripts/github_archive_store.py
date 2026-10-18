"""Upload immutable archives to GitHub Releases and stream-verify remote SHA256.

Never deletes local files. Credentials come from Git Credential Manager and are
used only for the configured GitHub repository, never written into receipts.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = 'zsure27/VLA-Quant-Archives'
TAG = 'backup-archive-20261017'
CHUNK = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text(encoding='utf-8'))
    assert plan['repository'] == REPO and plan['security_scan_passed']
    c = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\nusername=zsure27\n\n', text=True, capture_output=True, check=True)
    token = dict(x.split('=', 1) for x in c.stdout.splitlines() if '=' in x)['password']
    headers = {'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'VLA-archive-store', 'X-GitHub-Api-Version': '2022-11-28'}
    def api(path, data=None, method=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request('https://api.github.com/repos/'+REPO+('/'+path if path else ''), data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    assert api('')['private'] is True, 'Archive destination must remain private'
    try:
        release = api('releases/tags/'+TAG)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        commit = api('commits/main')['sha']
        release = api('releases', {'tag_name': TAG, 'target_commitish': commit, 'name': 'Research archive backup — 2026-10-17', 'body': 'Immutable experimental archives, raw data and source recovery bundles. Files are content-addressed; large files are split into 1 GiB parts. Recovery manifests and SHA256 indexes are maintained under backups/github in the repository. Existing experiment identities and denominators are preserved.', 'draft': False, 'prerelease': False, 'make_latest': 'false'})
    rid = release['id']
    assets = {}
    for page in range(1, 20):
        batch = api(f'releases/{rid}/assets?per_page=100&page={page}')
        assets.update({v['name']: v for v in batch})
        if len(batch) < 100:
            break
    state = json.loads(a.receipt.read_text()) if a.receipt.exists() else {'repository': REPO, 'release': release['html_url'], 'release_id': rid, 'tag': TAG, 'files': {}}
    def save():
        tmp = a.receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        tmp.replace(a.receipt)
    for entry in plan['files']:
        src = Path(entry['path'])
        key = entry['sha256']
        if state['files'].get(key, {}).get('remote_download_sha256_verified'):
            continue
        assert src.is_file() and src.stat().st_size == entry['bytes']
        whole = hashlib.sha256()
        parts = []
        count = (entry['bytes']+CHUNK-1)//CHUNK
        with src.open('rb') as f:
            for i in range(count):
                offset = i*CHUNK
                size = min(CHUNK, entry['bytes']-offset)
                ph = hashlib.sha256()
                f.seek(offset)
                remaining = size
                while remaining:
                    b = f.read(min(4*1024**2, remaining)); assert b
                    ph.update(b); whole.update(b); remaining -= len(b)
                digest = ph.hexdigest()
                name = 'sha256-'+digest+('.part%03d-of%03d' % (i+1, count))
                asset = assets.get(name)
                if not asset:
                    url = urllib.parse.urlsplit(release['upload_url'].split('{')[0]+'?name='+name)
                    con = http.client.HTTPSConnection(url.hostname, timeout=180)
                    con.putrequest('POST', url.path+'?'+url.query)
                    for k, v in {**headers, 'Content-Type': 'application/octet-stream', 'Content-Length': str(size)}.items():
                        con.putheader(k, v)
                    con.endheaders()
                    f.seek(offset); remaining = size; last = time.monotonic()
                    print(json.dumps({'upload': src.name, 'part': i+1, 'of': count, 'bytes': size}), flush=True)
                    while remaining:
                        b = f.read(min(1024**2, remaining)); assert b
                        con.send(b); remaining -= len(b)
                        if time.monotonic()-last >= 45:
                            print(json.dumps({'part_progress_percent': round(100*(size-remaining)/size, 1), 'asset': name}), flush=True); last = time.monotonic()
                    response = con.getresponse(); raw = response.read(); status = response.status; con.close()
                    if status != 201:
                        raise RuntimeError('GitHub upload failed HTTP '+str(status)+': '+raw.decode(errors='replace')[:500])
                    asset = json.loads(raw); assets[name] = asset
                assert asset['size'] == size and asset['state'] == 'uploaded'
                if asset.get('digest'):
                    assert asset['digest'] == 'sha256:'+digest
                rh = hashlib.sha256(); received = 0
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                        return None
                opener = urllib.request.build_opener(NoRedirect())
                req = urllib.request.Request(asset['url'], headers={**headers, 'Accept': 'application/octet-stream'})
                try:
                    response = opener.open(req, timeout=180)
                except urllib.error.HTTPError as e:
                    if e.code != 302:
                        raise
                    location = e.headers['Location']
                    host = urllib.parse.urlsplit(location).hostname
                    assert host in ('release-assets.githubusercontent.com', 'objects.githubusercontent.com')
                    response = urllib.request.urlopen(urllib.request.Request(location, headers={'User-Agent': 'VLA-archive-restore-check'}), timeout=180)
                with response:
                    while True:
                        b = response.read(4*1024**2)
                        if not b:
                            break
                        rh.update(b); received += len(b)
                assert received == size and rh.hexdigest() == digest
                parts.append({'asset_id': asset['id'], 'name': name, 'url': asset['browser_download_url'], 'offset': offset, 'bytes': size, 'sha256': digest, 'download_verified': True})
                print(json.dumps({'verified_part': i+1, 'of': count, 'file': src.name}), flush=True)
        assert whole.hexdigest() == key
        state['files'][key] = {'source_name': src.name, 'source_path': str(src), 'bytes': entry['bytes'], 'sha256': key, 'parts': parts, 'remote_download_sha256_verified': True}
        save()
        print(json.dumps({'verified_archive': src.name, 'bytes': entry['bytes'], 'sha256': key}), flush=True)
    save()
    print(json.dumps({'complete': True, 'archives': len(state['files']), 'release': state['release']}), flush=True)


if __name__ == '__main__':
    main()
