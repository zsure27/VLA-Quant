"""Private archive transport using system curl TLS, retaining v2 SHA contracts.

Secrets enter curl through stdin configuration, never argv or a config file.
Redirects are manual: API credentials are never sent to the asset CDN.
"""
import email.parser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import github_archive_store_v2 as core


class GitHubCurl(core.GitHub):
    def __init__(self, token, cache, curl):
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.curl = str(Path(curl).resolve(strict=True))
        super().__init__(token)

    def request(self, url, headers, *, method='GET', payload=None):
        name = uuid.uuid4().hex
        body, head = self.cache / (name + '.body'), self.cache / (name + '.headers')
        def quote(value):
            value = str(value)
            if '\r' in value or '\n' in value:
                raise core.ArchiveError('Newline in curl configuration rejected')
            return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
        options = ['silent', 'show-error', 'connect-timeout = 20', 'max-time = 180',
                   'proto = "=https"', 'request = ' + quote(method), 'url = ' + quote(url),
                   'dump-header = ' + quote(head.as_posix()), 'output = ' + quote(body.as_posix())]
        options += ['header = ' + quote(k + ': ' + v) for k, v in headers.items()]
        if payload is not None:
            options.append('data-binary = ' + quote('@' + Path(payload).as_posix()))
        result = subprocess.run([self.curl, '--config', '-'], input='\n'.join(options) + '\n',
                                text=True, capture_output=True, timeout=200)
        if result.returncode:
            raise ConnectionError('curl HTTPS request failed, exit ' + str(result.returncode))
        blocks = [b for b in head.read_text(encoding='iso-8859-1').strip().split('\n\n') if b.startswith('HTTP/')]
        if not blocks:
            raise core.ArchiveError('No HTTP response headers')
        lines = blocks[-1].splitlines()
        status = int(lines[0].split()[1])
        response_headers = email.parser.Parser().parsestr('\n'.join(lines[1:]))
        return status, response_headers, body

    def api(self, path):
        url = 'https://api.github.com/repos/' + core.REPO + ('/' + path if path else '')
        core.safe_url(url, {'api.github.com'}, '/repos/' + core.REPO)
        def fetch():
            self.budget.acquire()
            status, headers, body = self.request(url, self.headers)
            self.budget.observe(headers)
            if status != 200:
                raise core.HTTPFailure(status, headers)
            return json.loads(body.read_text(encoding='utf-8'))
        return core.with_retries(fetch)

    def upload(self, part):
        core.source_unchanged(part)
        payload = part.source
        if part.offset or part.size != part.file_bytes:
            payload = self.cache / (part.sha256 + '.upload-chunk')
            if not payload.exists():
                with part.source.open('rb') as source, payload.open('xb') as target:
                    source.seek(part.offset)
                    remaining = part.size
                    while remaining:
                        block = source.read(min(core.READ_SIZE, remaining))
                        if not block:
                            raise core.ArchiveError('Truncated source chunk')
                        target.write(block)
                        remaining -= len(block)
        h = hashlib.sha256()
        with payload.open('rb') as stream:
            for block in iter(lambda: stream.read(core.READ_SIZE), b''):
                h.update(block)
        if payload.stat().st_size != part.size or h.hexdigest() != part.sha256:
            raise core.HashMismatch('Curl payload SHA differs')
        url = self.release['upload_url'].split('{')[0] + '?name=' + urllib.parse.quote(part.name)
        core.safe_url(url, {'uploads.github.com'}, '/repos/' + core.REPO + '/releases/%s/assets' % self.release['id'])
        self.budget.acquire()
        status, headers, body = self.request(url, dict(self.headers, **{'Content-Type': 'application/octet-stream'}), method='POST', payload=payload)
        self.budget.observe(headers)
        if status != 201:
            raise core.HTTPFailure(status, headers)
        asset = json.loads(body.read_text(encoding='utf-8'))
        with self.asset_lock:
            self.assets[part.name] = asset
        return asset

    def verify_download(self, asset, part):
        url = 'https://api.github.com/repos/' + core.REPO + '/releases/assets/' + str(asset['id'])
        headers = dict(self.headers, Accept='application/octet-stream')
        for redirect in range(4):
            core.safe_url(url, {'api.github.com'} if redirect == 0 else core.DOWNLOAD_HOSTS,
                          '/repos/' + core.REPO if redirect == 0 else None)
            self.budget.acquire()
            status, response_headers, body = self.request(url, headers)
            if redirect == 0:
                self.budget.observe(response_headers)
            if status in (301, 302, 303, 307, 308):
                url = response_headers.get('Location', '')
                core.safe_url(url, core.DOWNLOAD_HOSTS)
                headers = {'User-Agent': 'VLA-private-restore-SHA-v1'}
                continue
            if status != 200:
                raise core.HTTPFailure(status, response_headers)
            h = hashlib.sha256()
            with body.open('rb') as stream:
                for block in iter(lambda: stream.read(core.READ_SIZE), b''):
                    h.update(block)
            if body.stat().st_size != part.size or h.hexdigest() != part.sha256:
                raise core.HashMismatch('Curl private download SHA differs')
            body.unlink()
            return
        raise core.ArchiveError('Too many private asset redirects')
