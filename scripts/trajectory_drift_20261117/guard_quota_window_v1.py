"""Validate actual quota and allow only the preregistered bounded refresh wait.

WAIT_API_REFRESH authorizes no work: callers must pause their own active groups.
Missing/stale observations and <=10% never receive the refresh wait exemption.
"""
from datetime import datetime, timezone
import math


def bounded_quota_decision(observation, control, current=None):
    current = current or datetime.now(timezone.utc)
    if observation.get('source') != 'codex_api' or observation.get('window_minutes') != 300:
        raise ValueError('Actual five-hour API observation required')
    age = (current-datetime.fromisoformat(observation['observed_at'].replace('Z','+00:00'))).total_seconds()
    if not 0 <= age <= control.get('quota_max_age_seconds',1020):
        raise ValueError('Quota unavailable/stale; save and close')
    remaining, reset = observation['remaining_percent'], observation['resets_at']
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in [remaining,reset]):
        raise ValueError('Nonfinite or invalid quota')
    if not 0 <= remaining <= 100 or reset > current.timestamp()+18300:
        raise ValueError('Invalid actual quota window')
    if remaining <= control['close_remaining_percent']:
        return 'CLOSE'
    if reset <= current.timestamp():
        wait = control.get('refresh_confirmation_wait_seconds',0)
        if type(wait) is not int or not 0 <= wait <= 120:
            raise ValueError('Refresh wait must be bounded to 0..120 seconds')
        return 'WAIT_API_REFRESH' if wait and current.timestamp()-reset <= wait else 'CLOSE'
    if remaining <= control['stop_expansion_remaining_percent']:
        return 'STOP_EXPANSION'
    return 'CONTINUE'
