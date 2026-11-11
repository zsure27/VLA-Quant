"""Correct 40-character GR00T snapshot revisions, with unchanged full member SHA checks.

The frozen 20261108 multi-suite constants contain three 41-character transcription
errors. The valid identities below match the original download receipts and the
20261109 complete 29-file CPU re-audit; no source receipt is edited.
"""
import json
from pathlib import Path
import re

from qvla.execute_groot_bf16_episode_v2 import sha

SUITES={
 'libero_spatial':('groot_spatial','youliangtan/gr00t-n1.5-libero-spatial-posttrain','03294ae3de7d4870be8ae6e27cbdeffcb46b6425',220),
 'libero_object':('groot_object','youliangtan/gr00t-n1.5-libero-object-posttrain','f422266b016fd917127d3a0348062a5cd750df43',280),
 'libero_goal':('groot_goal','youliangtan/gr00t-n1.5-libero-goal-posttrain','17be45f5bf8a543594761b263c68cf33f33847bd',600),
 'libero_10':('groot_long','youliangtan/gr00t-n1.5-libero-long-posttrain','aa49078d5cc9ce72917bc4312f1ef12771f277de',1000),
}
assert all(re.fullmatch('[0-9a-f]{40}',x[2]) for x in SUITES.values())


def verify_snapshot(checkpoint:Path,suite:str):
    name,repo,revision,_=SUITES[suite]
    if checkpoint.name!=name or checkpoint.is_symlink():raise ValueError('Wrong or symlinked checkpoint')
    data=json.loads((checkpoint/'DOWNLOAD_VERIFICATION.json').read_text())
    rows=data.get('files_verified')
    if data.get('repo')!=repo or data.get('revision')!=revision or not isinstance(rows,list) or len(rows)!=data.get('files'):
        raise ValueError('Checkpoint identity/source receipt differs')
    total=0;seen=set()
    for row in rows:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen:raise ValueError('Unsafe or duplicate member')
        seen.add(relative.as_posix());item=checkpoint/relative
        if item.is_symlink() or not item.is_file() or item.stat().st_size!=row['bytes'] or sha(item)!=row['sha256']:
            raise ValueError('Checkpoint source hash differs: '+str(relative))
        total+=row['bytes']
    if total!=data['bytes']:raise ValueError('Checkpoint size sum differs')
    return data
