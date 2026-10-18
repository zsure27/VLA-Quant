"""Publish small recovery indexes to the verified private archive repository.

Large data belongs in Release assets. This commits only explicitly selected
files, verifies the resulting remote blobs, and never changes visibility.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import urllib.error
import urllib.request

REPO = 'zsure27/VLA-Quant-Archives'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_files(selected):
    """Reject ambiguous remote paths before creating any Git objects."""
    require(selected.get('repository') == REPO, 'Unexpected private repository')
    require(isinstance(selected.get('files'), list) and selected['files'], 'No selected files')
    prepared = []
    targets = set()
    for entry in selected['files']:
        raw_target = entry['target']
        require(isinstance(raw_target, str) and raw_target and '\\' not in raw_target,
                'Target must use an explicit POSIX repository path')
        target = PurePosixPath(raw_target)
        require(not target.is_absolute() and '..' not in target.parts and
                str(target) == raw_target, 'Unsafe or noncanonical repository path')
        require(str(target) not in targets, 'Duplicate publication target: ' + str(target))
        targets.add(str(target))
        require(target.suffix in ('.md', '.json', '.py', '.ps1'), 'Unsupported index file type')
        raw = Path(entry['source']).read_bytes()
        require(len(raw) < 10*1024**2, 'Large files must use Release assets')
        prepared.append((str(target), raw))
    return prepared


def verify_committed_paths(api, commit_sha, tree_items):
    """Bind every published target to the exact blob in the committed tree."""
    commit = api('git/commits/' + commit_sha)
    require(commit['sha'] == commit_sha, 'Remote commit identity mismatch')
    tree = api('git/trees/' + commit['tree']['sha'] + '?recursive=1')
    require(tree.get('truncated') is False, 'Committed tree listing was truncated')
    by_path = {item['path']: item for item in tree['tree']}
    for expected in tree_items:
        actual = by_path.get(expected['path'], {})
        require(actual.get('type') == 'blob' and actual.get('mode') == expected['mode'] and
                actual.get('sha') == expected['sha'],
                'Committed target does not reference the expected blob: ' + expected['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files-map', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--message', required=True)
    args = parser.parse_args()
    selected = json.loads(args.files_map.read_text(encoding='utf-8'))
    prepared = prepare_files(selected)
    result = subprocess.run(['git', 'credential', 'fill'],
                            input='protocol=https\nhost=github.com\nusername=zsure27\n\n',
                            text=True, capture_output=True, check=True)
    token = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)['password']
    headers = {'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json',
               'User-Agent': 'VLA-private-archive-index', 'X-GitHub-Api-Version': '2022-11-28'}

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **kw):
            return None

    opener = urllib.request.build_opener(NoRedirect())

    def api(path, data=None, method=None):
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request('https://api.github.com/repos/'+REPO+('/'+path if path else ''),
                                         data=body, headers=headers, method=method)
        try:
            with opener.open(request, timeout=90) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError('Private index API failed HTTP '+str(error.code)) from None

    repository = api('')
    require(repository['private'] is True and repository['full_name'] == REPO,
            'Publication destination must be the expected private repository')
    branch = repository['default_branch']
    before = api('git/ref/heads/'+branch)['object']['sha']
    tree_before = api('git/commits/'+before)['tree']['sha']
    tree_items = []
    for target, raw in prepared:
        blob = api('git/blobs', {'content': base64.b64encode(raw).decode(), 'encoding': 'base64'})
        require(blob['sha'] == hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                'Uploaded Git blob SHA mismatch')
        tree_items.append({'path': target, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    tree = api('git/trees', {'base_tree': tree_before, 'tree': tree_items})
    commit = api('git/commits', {'message': args.message, 'tree': tree['sha'], 'parents': [before]})
    verify_committed_paths(api, commit['sha'], tree_items)
    # No force: a concurrent branch update cannot be silently discarded.
    api('git/refs/heads/'+branch, {'sha': commit['sha'], 'force': False}, 'PATCH')
    require(api('git/ref/heads/'+branch)['object']['sha'] == commit['sha'],
            'Remote branch does not point to the published commit')
    verified = []
    for item, (target, raw) in zip(tree_items, prepared):
        stored = api('git/blobs/'+item['sha'])
        remote = base64.b64decode(stored['content'])
        require(remote == raw, 'Remote Git blob bytes differ from the selected local index')
        verified.append({'path': target, 'bytes': len(raw), 'sha256': hashlib.sha256(remote).hexdigest(),
                         'git_blob_sha': item['sha']})
    receipt = {'repository': REPO, 'private': True, 'branch': branch, 'commit': commit['sha'],
               'remote_head_verified': True, 'committed_paths_verified': True,
               'files': verified}
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'repository': REPO, 'commit': commit['sha'], 'verified_files': len(verified)}))


if __name__ == '__main__':
    main()
