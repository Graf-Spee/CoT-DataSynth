"""Metered OpenAI-compatible synthesis client; no provider SDK dependency.

Plugins import complete(context, messages, **sampling). Credentials stay in env.
This client accounts for its own requests; it is not a network isolation boundary.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from .common import Store


def complete(context, messages, **sampling):
    teacher = context['teacher_api']
    base = os.environ.get(teacher.get('base_url_env', ''), teacher.get('base_url', ''))
    if not base:
        raise ValueError('teacher base URL is not configured')
    key = os.environ.get(teacher['api_key_env'])
    if not key:
        raise ValueError('teacher API key is not configured')
    store = Store(context['session'])
    maximum = int(sampling.get('max_tokens', 2048))
    if maximum <= 0:
        raise ValueError('max_tokens must be positive')
    # Reserve conservatively, including prompt bytes as an upper-bound proxy.
    reservation = maximum + len(json.dumps(messages).encode('utf-8'))
    limit = context['budget'].get('max_api_tokens', 0)
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        spent = store.db.execute('SELECT COALESCE(SUM(tokens),0) FROM usage WHERE category IN '
                                 "('synthesis_api','api_reservation')").fetchone()[0]
        if limit and spent + reservation > limit:
            raise ValueError('synthesis API token budget exhausted')
        cursor = store.db.execute('INSERT INTO usage(attempt,category,tokens,created,detail) VALUES(?,?,?,?,?)',
                                  (context['attempt_id'], 'api_reservation', reservation, time.time(), '{}'))
        reservation_id = cursor.lastrowid
    payload = {'model': teacher['model'], 'messages': messages, 'max_tokens': maximum, **sampling}
    request = urllib.request.Request(base.rstrip('/') + '/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=teacher.get('timeout_s', 180)) as response:
            result = json.load(response)
    except Exception:
        # Retain reservation: server may have consumed tokens before connection failed.
        raise
    usage = result.get('usage')
    if usage:
        tokens = int(usage['total_tokens'])
        cost = None
        if 'input_usd_per_million' in teacher and 'output_usd_per_million' in teacher:
            cost = (usage['prompt_tokens'] * teacher['input_usd_per_million'] +
                    usage['completion_tokens'] * teacher['output_usd_per_million']) / 1e6
        with store.db:
            store.db.execute('UPDATE usage SET category=?,tokens=?,cost=?,detail=? WHERE id=?',
                             ('synthesis_api', tokens, cost, json.dumps(usage), reservation_id))
    return result
