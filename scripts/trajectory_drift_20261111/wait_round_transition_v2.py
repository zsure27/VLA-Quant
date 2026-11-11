"""Current-task event wait only; terminal initial status returns immediately.

This is not an app automation and does not fetch API quota after the controller
turn ends. A timeout asks the controller for its 15-minute heartbeat.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import time


def changed(value,initial):
    return value['state'] not in ['RUNNING','PREFLIGHT'] or value.get('current_stage')!=initial.get('current_stage')


def wait_transition(root,maximum=800):
    def read():return json.loads((root/'status.json').read_text())
    initial=read()
    # The v1 local observer missed this check and could wait on an already failed round.
    if changed(initial,initial):return dict(event='ALREADY_TERMINAL',status=initial)
    library=ctypes.CDLL(None);fd=library.inotify_init1(0)
    if fd<0:raise OSError('inotify unavailable; do not claim an active event monitor')
    try:
        if library.inotify_add_watch(fd,os.fsencode(root),0x80|0x8)<0:raise OSError('Cannot watch round directory')
        # Close the read-to-watch registration race.
        current=read()
        if changed(current,initial):return dict(event='STAGE_CHANGED',status=current)
        start=time.monotonic()
        while time.monotonic()-start<maximum:
            ready,_,_=select.select([fd],[],[],min(50,maximum-(time.monotonic()-start)))
            if ready:
                os.read(fd,65536);current=read()
                if changed(current,initial):return dict(event='STAGE_CHANGED',status=current)
        return dict(event='HEARTBEAT_DUE',status=read())
    finally:os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('round_dir',type=Path);p.add_argument('--maximum-seconds',type=int,default=800)
    a=p.parse_args()
    if not 1<=a.maximum_seconds<=900:raise ValueError('Bound event wait to the heartbeat interval')
    print(json.dumps(wait_transition(a.round_dir,a.maximum_seconds)),flush=True)
