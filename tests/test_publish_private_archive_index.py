"""Offline proof that an incremental publication retains only unchanged blobs."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import publish_private_archive_index as publisher


class IncrementalPublicationTests(unittest.TestCase):
    def setUp(self):
        self.old = {'path': 'catalog/old.json', 'bytes': 3,
                    'sha256': hashlib.sha256(b'old').hexdigest(), 'git_blob_sha': 'a' * 40}
        self.changed = {'path': 'catalog/current.json', 'bytes': 3,
                        'sha256': hashlib.sha256(b'v1!').hexdigest(), 'git_blob_sha': 'b' * 40}
        self.previous = {'repository': publisher.REPO, 'private': True,
                         'remote_head_verified': True, 'committed_paths_verified': True,
                         'commit': 'c' * 40, 'files': [self.old, self.changed]}
        self.new_commit = 'd' * 40

    def verify(self, inherited, changed_old_blob=False):
        new = {'path': 'catalog/current.json', 'type': 'blob', 'mode': '100644', 'sha': 'e' * 40}
        old = {'path': self.old['path'], 'type': 'blob', 'mode': '100644',
               'sha': 'f' * 40 if changed_old_blob else self.old['git_blob_sha']}
        responses = {'git/commits/' + self.new_commit:
                     {'sha': self.new_commit, 'tree': {'sha': '1' * 40}},
                     'git/trees/' + '1' * 40 + '?recursive=1':
                     {'truncated': False, 'tree': [new, old]}}
        expected = [new] + [{'path': item['path'], 'type': 'blob', 'mode': '100644',
                            'sha': item['git_blob_sha']} for item in inherited]
        publisher.verify_committed_paths(responses.__getitem__, self.new_commit, expected)

    def test_unchanged_catalog_inherited_while_selected_path_is_freshly_verified(self):
        inherited = publisher.inherited_files(self.previous, {'catalog/current.json'})
        self.assertEqual(inherited, [self.old])
        self.verify(inherited)
        self.assertNotIn(self.changed, inherited)

    def test_inherited_blob_changed_in_new_commit_rejected(self):
        inherited = publisher.inherited_files(self.previous, {'catalog/current.json'})
        with self.assertRaisesRegex(ValueError, 'Committed target'):
            self.verify(inherited, changed_old_blob=True)

    def test_previous_unverified_publication_rejected_before_inheritance(self):
        self.previous['committed_paths_verified'] = False
        with self.assertRaises(ValueError):
            publisher.inherited_files(self.previous, {'catalog/current.json'})


if __name__ == '__main__':
    unittest.main()
