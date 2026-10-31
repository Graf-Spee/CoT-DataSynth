from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path

from .common import REPO, finite, read_json, write_json


def eval_summary(directory):
    directory = Path(directory)
    # Our Harbor adapter and fixed custom adapters use this normalized contract.
    normalized = directory / 'summary.json'
    if normalized.is_file():
        result = read_json(normalized)
        if not finite(result.get('score')) or not 0 <= result['score'] <= 1:
            raise ValueError(f'invalid evaluation score: {normalized}')
        if result.get('n_tasks', 0) <= 0:
            raise ValueError('evaluation must include n_tasks > 0')
        return result
    candidates = sorted(directory.rglob('results.json'))
    for path in candidates:
        result = read_json(path)
        if finite(result.get('accuracy')):
            if not 0 <= result['accuracy'] <= 1:
                raise ValueError('accuracy must be a fraction in [0,1]')
            count = result.get('num_samples', result.get('num_prompts', 0))
            if not finite(count) or count <= 0:
                raise ValueError('evaluation must contain at least one task')
            return {'score': result['accuracy'], 'n_tasks': result.get('num_samples', result.get('num_prompts')),
                    'raw': str(path), 'details': result}
    raise ValueError(f'evaluation produced no recognized metrics under {directory}')


def curves(attempt_dir):
    """Export real numeric log entries; SVG avoids a matplotlib dependency."""
    import sys
    sys.path.insert(0, str(REPO / 'DataObs' / 'tools'))
    from grpo_log_metrics import clean_line, PAIR_RE, STEP_RE
    output = Path(attempt_dir) / 'metrics'
    output.mkdir(exist_ok=True)
    all_series = {}
    for stage in ('sft', 'rl'):
        logs = list((Path(attempt_dir) / 'experiment' / 'logs').glob('*.log'))
        logs += list((Path(attempt_dir) / 'logs').glob('*.log'))
        logs = [p for p in logs if p.exists() and
                (p.stem == stage or (stage == 'rl' and p.stem == 'grpo'))]
        points = []
        for path in logs:
            for line in path.read_text(errors='replace').splitlines():
                line = clean_line(line)
                match = STEP_RE.search(line)
                pairs = {m['key']: float(m['value']) for m in PAIR_RE.finditer(line)
                         if math.isfinite(float(m['value']))}
                if not pairs:
                    # SFT console logger may serialize Python dict entries.
                    pairs = {k: float(v) for k, v in re.findall(
                        r"['\"]([\w/.@-]+)['\"]:\s*([+-]?[\d.]+(?:e[+-]?\d+)?)", line)}
                if pairs:
                    points.append({'step': int(match['step']) if match else len(points), **pairs})
        if not points:
            all_series[stage] = {'available': False, 'reason': 'no structured numeric log entries'}
            continue
        fields = sorted(set().union(*(p.keys() for p in points)))
        with (output / f'{stage}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(points)
        with (output / f'{stage}.jsonl').open('w') as stream:
            for point in points:
                stream.write(json.dumps(point) + '\n')
        series = [k for k in fields if any(s in k for s in ('loss', 'reward', 'entropy', 'kl', 'grad_norm'))]
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="%s">' % (max(1, len(series)) * 180)]
        for index, key in enumerate(series):
            values = [(p['step'], p[key]) for p in points if key in p and finite(p[key])]
            if not values:
                continue
            xs, ys = zip(*values)
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            coords = ' '.join(f'{45+710*(x-xmin)/(xmax-xmin or 1):.2f},'
                              f'{index*180+150-110*(y-ymin)/(ymax-ymin or 1):.2f}' for x,y in values)
            svg += [f'<text x="20" y="{index*180+22}">{html.escape(key)} ({ymin:.4g} .. {ymax:.4g})</text>',
                    f'<polyline points="{coords}" fill="none" stroke="#2563eb"/>']
        svg.append('</svg>')
        (output / f'{stage}.svg').write_text('\n'.join(svg))
        all_series[stage] = {'available': True, 'points': len(points), 'metrics': series}
    write_json(output / 'index.json', all_series)
    return all_series


def build_feedback(store, attempt, cfg):
    root = store.session / 'runner' / 'attempts' / attempt
    sft, rl = (eval_summary(root / 'eval' / name) for name in ('sft', 'rl'))
    recipe = read_json(store.session / 'runner' / 'snapshots' / attempt / 'recipe.json')
    feedback = {'schema_version': 1, 'attempt_id': attempt, 'hypothesis': recipe['hypothesis'],
                'parent_attempt_id': recipe.get('parent_attempt_id'), 'recipe': recipe,
                'objective': 'rl_dev_score', 'sft': sft, 'rl': rl,
                'sft_to_rl_gain': rl['score'] - sft['score'],
                'dataset': read_json(root / 'dataset' / 'summary.json'),
                'curves': curves(root), 'probe': read_json(root / 'probes' / 'summary.json'),
                'usage': [r for r in store.costs() if r['attempt'] == attempt],
                'cost_note': 'Unpriced categories remain unknown; known_cost_usd is not total cost.',
                'artifacts': str(root), 'base_model': cfg['base_model']}
    if cfg['training'].get('agentic_rl'):
        traces = list((root / 'agentic/rl').glob('*/trajectory.json'))
        episodes = [read_json(p) for p in traces]
        accounting = [read_json(p) for p in (root / 'agentic').glob('*/*/episode.json')]
        feedback['agentic'] = {'rollouts': len(episodes),
            'mean_reward_by_split': {split: sum(e['reward'] for e in episodes if e['split'] == split) /
                max(1, sum(e['split'] == split for e in episodes)) for split in ('train', 'val')},
            'termination_counts': {reason: sum(e['termination'] == reason for e in episodes)
                                   for reason in sorted({e['termination'] for e in episodes})},
            'sandbox_elapsed_s': sum(e['elapsed_s'] for e in accounting),
            'infrastructure_failures': sum(e['result'] is None or bool(e['cleanup_error']) for e in accounting),
            'trajectories': str(root / 'agentic'), 'cost_usd': None}
    write_json(root / 'feedback.json', feedback)
    return feedback
