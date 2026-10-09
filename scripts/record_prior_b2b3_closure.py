"""Verify the old archive on each host and issue a SHA-bound B4 closure receipt."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qvla.extended_peft import sha


def verify(archive):
    archive=archive.resolve(strict=True)
    counts={}; hashes={}
    for name, expected_count in (("SHA256SUMS.txt",13),("RESULTS_SHA256SUMS.txt",18494)):
        lines=(archive/name).read_text(encoding="utf-8").splitlines()
        if len(lines)!=expected_count: raise ValueError(f"Incomplete {name}")
        for line in lines:
            expected, relative=line.split(maxsplit=1)
            path=(archive/relative.lstrip("*")).resolve(strict=True)
            if archive not in path.parents or sha(path)!=expected:
                raise ValueError(f"Archive SHA/path mismatch: {relative}")
        counts[name]=len(lines); hashes[name]=sha(archive/name)
    scope=json.loads((archive/"SERVER_ARCHIVE_SCOPE.json").read_text(encoding="utf-8"))
    if not (archive/f"VLA-Quant-{scope['server_commit']}.bundle").is_file():
        raise ValueError("Missing original Git bundle")
    return {"archive":str(archive),"verified_counts":counts,"manifest_sha256":hashes,
            "server_commit":scope["server_commit"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--server",action="store_true")
    p.add_argument("--expected-hostname")
    p.add_argument("--server-verification",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError("Closure receipt exists; do not overwrite")
    record=verify(a.archive)
    if a.server:
        if socket.gethostname()!=a.expected_hostname: raise ValueError("Wrong active server hostname")
        record.update(gate="PASS_PERSISTENT_ARCHIVE",verified_hostname=socket.gethostname())
    else:
        if a.server_verification is None: p.error("Local receipt requires --server-verification")
        server=json.loads(a.server_verification.read_text(encoding="utf-8"))
        if server.get("gate")!="PASS_PERSISTENT_ARCHIVE" or not server.get("verified_hostname") or any(
                server.get(k)!=record[k] for k in ("verified_counts","manifest_sha256","server_commit")):
            raise ValueError("Verified server/local archives differ")
        git=["git","-c",f"safe.directory={ROOT}","-C",str(ROOT),"-c","credential.username=zsure27",
             "-c","credential.interactive=false"]
        remote=subprocess.check_output([*git,"remote","get-url","origin"],text=True).strip()
        if remote not in ("https://github.com/zsure27/VLA-Quant.git","https://zsure27@github.com/zsure27/VLA-Quant.git"):
            raise ValueError("Wrong GitHub target")
        env=dict(os.environ,GIT_TERMINAL_PROMPT="0",GCM_INTERACTIVE="never")
        head=subprocess.check_output([*git,"rev-parse","HEAD"],text=True).strip()
        # On this Windows host ls-remote can fail its Schannel handshake even when
        # the authenticated fetch succeeds. FETCH_HEAD is the actual advertised
        # main commit fetched in this invocation, not a stale tracking ref.
        subprocess.check_call([*git,"fetch","origin","main"],env=env,timeout=60)
        advertised=subprocess.check_output([*git,"rev-parse","FETCH_HEAD"],text=True,env=env).strip()
        if advertised!=head: raise ValueError("Current report commit not synchronized to GitHub")
        record={"gate":"PASS_PRIOR_CLOSURE","server_gate":server["gate"],"local_gate":"PASS_LOCAL_ARCHIVE_SHA256",
            "server_archive":server["archive"],"local_archive":record["archive"],"verified_counts":record["verified_counts"],
            "server_manifest_sha256":server["manifest_sha256"],"local_manifest_sha256":record["manifest_sha256"],
            "server_verification_sha256":sha(a.server_verification),"github_repository":"zsure27/VLA-Quant",
            "verified_remote_sha":head,"server_commit":server["server_commit"],
            "note":"Archive and Git receipt; prior 046 shutdown remains user-reported, no independent platform OFF claim"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"gate":record["gate"],"output":str(a.output)}))


if __name__=="__main__": main()
