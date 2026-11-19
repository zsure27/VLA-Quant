"""User-selected normal closure policy; missing evidence pauses, never guesses quota."""
import datetime
import math


def decision(quota, *, now=None, authorization_valid=True):
    now=now or datetime.datetime.now(datetime.timezone.utc)
    try:
        observed=datetime.datetime.fromisoformat(quota['observed_at'].replace('Z','+00:00'))
        age=(now-observed).total_seconds()
        value=quota['remaining_percent']
        valid=(quota['source']=='codex_api' and quota['window_minutes']==300
               and 0<=age<=1020 and type(value) in (int,float) and math.isfinite(value)
               and 0<=value<=100 and type(quota['resets_at']) is int
               and quota['resets_at']>now.timestamp())
    except (KeyError,TypeError,ValueError,AttributeError):valid=False
    if not valid:return 'PAUSE_SAVE_ALERT_PRIMARY_UNAVAILABLE'
    if value<=10:return 'CLOSE_FRESH_PRIMARY_THRESHOLD'
    if not authorization_valid:return 'PAUSE_SAVE_ALERT_AUTHORIZATION'
    if value<=15:return 'STOP_EXPANSION_SAVE'
    return 'CONTINUE'
