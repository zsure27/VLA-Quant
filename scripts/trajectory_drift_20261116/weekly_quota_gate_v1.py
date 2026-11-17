"""Current-run user rule: actual weekly remaining <=3% starts closure."""
from datetime import datetime,timezone
import math


def weekly_close_required(observation,current=None):
    current=current or datetime.now(timezone.utc)
    if observation.get('source')!='codex_api' or observation.get('window_minutes')!=10080:
        raise ValueError('Actual weekly API observation required')
    age=(current-datetime.fromisoformat(observation['observed_at'].replace('Z','+00:00'))).total_seconds()
    if not 0<=age<=1020:
        raise ValueError('Weekly quota observation unavailable/stale')
    remaining,reset=observation['remaining_percent'],observation['resets_at']
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in [remaining,reset]):
        raise ValueError('Invalid weekly quota values')
    if not 0<=remaining<=100 or not current.timestamp()<reset<=current.timestamp()+605100:
        raise ValueError('No inferred weekly refresh allowed')
    return remaining<=3
