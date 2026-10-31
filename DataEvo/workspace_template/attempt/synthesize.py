"""Runnable GSM8K API baseline; Codex is free to replace this program."""
import argparse
import json
from pathlib import Path

from DataEvo.runtime.api import complete
from DataEvo.runtime.data import load_rows

parser = argparse.ArgumentParser()
parser.add_argument('--context', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
context = json.loads(Path(args.context).read_text())
generation = context['generation']
template = Path(generation.get('template', 'teacher.txt')).read_text()
rows = load_rows(context['data']['seed'])[:generation.get('max_tasks', 64)]
with Path(args.output).open('w') as stream:
    for index, row in enumerate(rows):
        question = row.get('question')
        if question is None:
            question = next(m['content'] for m in row['prompt'] if m['role'] == 'user')
        gold = row.get('answer')
        if gold is None:
            gold = row.get('reward_model', {}).get('ground_truth')
        if gold is None:
            raise ValueError('seed needs answer or reward_model.ground_truth')
        for sample in range(generation.get('samples_per_task', 2)):
            result = complete(context, [{'role': 'user', 'content': template.format(question=question)}],
                              **generation.get('sampling', {}))
            output = {'task_id': f'seed-{index}', 'question': question,
                      'answer': result['choices'][0]['message']['content'], 'gold': str(gold),
                      'sample_index': sample, 'source': context['data']['seed'],
                      'finish_reason': result['choices'][0].get('finish_reason')}
            stream.write(json.dumps(output, ensure_ascii=False) + '\n')
