"""Known-answer correctness + exact dedup baseline, with per-row audit reasons."""
import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from DataEvo.runtime.data import load_rows


def numeric(text):
    tail = str(text).rsplit('####', 1)[-1].replace(',', '')
    numbers = re.findall(r'-?\d+(?:\.\d+)?', tail)
    try:
        return Decimal(numbers[-1]) if numbers else None
    except InvalidOperation:
        return None


parser = argparse.ArgumentParser()
parser.add_argument('--context', required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
context = json.loads(Path(args.context).read_text())
seen = set()
with Path(args.output).open('w') as output, (Path(context['artifact_dir']) / 'selection.jsonl').open('w') as audit:
    for index, row in enumerate(load_rows(args.input)):
        signature = (row['question'].strip(), row['answer'].strip())
        expected = numeric(row.get('gold', ''))
        reason = 'keep'
        if row.get('finish_reason') == 'length':
            reason = 'truncated'
        elif '####' not in row['answer'] or expected is None or numeric(row['answer']) != expected:
            reason = 'incorrect_or_unknown_gold'
        elif signature in seen:
            reason = 'duplicate'
        seen.add(signature)
        audit.write(json.dumps({'row': index, 'task_id': row.get('task_id'), 'reason': reason}) + '\n')
        if reason == 'keep':
            row['verified'] = True
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
