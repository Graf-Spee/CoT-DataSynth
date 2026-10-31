from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGES = ('generate', 'postprocess', 'validate_data', 'sft', 'sft_eval', 'rl', 'rl_eval', 'probe', 'feedback')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.writing-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('attempt id must contain only letters, digits, _ and - (max 80)')
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_manifest(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def within(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'path escapes attempt: {relative}')
    return path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Store:
    def __init__(self, session):
        self.session = Path(session).resolve()
        self.db = sqlite3.connect(self.session / 'runner' / 'session.sqlite', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, state TEXT NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, error TEXT, pid INTEGER, score REAL);
            CREATE TABLE IF NOT EXISTS stages (
                attempt TEXT, name TEXT, state TEXT, started REAL, ended REAL,
                PRIMARY KEY(attempt,name));
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY, attempt TEXT, category TEXT, tokens INTEGER,
                cost REAL, created REAL, detail TEXT);
        ''')

    def rows(self):
        return [dict(row) for row in self.db.execute('SELECT * FROM attempts ORDER BY created')]

    def get(self, attempt):
        row = self.db.execute('SELECT * FROM attempts WHERE id=?', (identifier(attempt),)).fetchone()
        if row is None:
            raise ValueError(f'unknown attempt: {attempt}')
        return dict(row)

    def update(self, attempt, state, error=None, score=None):
        with self.db:
            self.db.execute('UPDATE attempts SET state=?,updated=?,error=?,score=COALESCE(?,score) WHERE id=?',
                            (state, time.time(), error, score, attempt))

    def usage(self, attempt, category, tokens=0, cost=None, detail=None):
        with self.db:
            self.db.execute('INSERT INTO usage(attempt,category,tokens,cost,created,detail) VALUES(?,?,?,?,?,?)',
                            (attempt, category, tokens, cost, time.time(), json.dumps(detail or {})))

    def costs(self):
        return [dict(r) for r in self.db.execute(
            'SELECT attempt,category,SUM(tokens) tokens,SUM(cost) known_cost_usd,COUNT(*) entries '
            'FROM usage GROUP BY attempt,category')]


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def config(session):
    return read_json(Path(session) / 'runner' / 'config.json')
