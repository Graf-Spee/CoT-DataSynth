"""Evaluation contracts; HTTP is real, sandbox/Harbor execution doubles are explicit."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from DataEvo.runtime.benchmark_eval import (
    build_config, collect_trials, execute, make_plan, mini_config, preflight, summarize_benchmark,
)
from DataEvo.runtime.common import write_json
from DataEvo.scripts.evaluate_agentic_benchmarks import main, parser


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        for name in ('b-task', 'a-task'):
            task = self.dataset / name
            (task / 'environment').mkdir(parents=True)
            (task / 'tests').mkdir()
            (task / 'solution').mkdir()
            (task / 'instruction.md').write_text('Solve a fixture task; no real benchmark here.')
            (task / 'task.toml').write_text('version = "1.0"\n')
            (task / 'environment/Dockerfile').write_text('FROM ubuntu:24.04\n')
            (task / 'tests/test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
            (task / 'solution/solve.sh').write_text('FIXTURE_SOLUTION_NOT_FOR_MODEL')

    def tearDown(self):
        self.temp.cleanup()

    def args(self, *extra):
        return parser().parse_args(['--benchmark', 'terminal_bench_2', '--terminal-dataset',
            str(self.dataset), '--model', 'fixture-model', *extra])

    def raw(self):
        return {'n_total_trials': 2, 'stats': {'n_completed_trials': 2, 'n_errored_trials': 0,
                'evals': {'fixture': {'reward_stats': {'reward': {'1': ['a'], '0': ['b']}}}}}}

    def trial(self, name, score):
        return {'task_name': name, 'trial_name': name + '__fixture',
                'finished_at': '2026-10-19T00:00:00Z', 'verifier_result': {'rewards': {'reward': score}}}

    def test_selection_identity_and_missing_task(self):
        plan = make_plan(self.args())
        self.assertEqual([t['id'] for t in plan['benchmarks'][0]['tasks']], ['a-task'])
        self.assertNotIn('FIXTURE_SOLUTION_NOT_FOR_MODEL', json.dumps(plan))
        before = plan['benchmarks'][0]['tasks'][0]['files']['tests/test.sh']
        (self.dataset / 'a-task/tests/test.sh').write_text('changed verifier')
        after = make_plan(self.args())['benchmarks'][0]['tasks'][0]['files']['tests/test.sh']
        self.assertNotEqual(before, after)
        self.assertEqual(make_plan(self.args('--task', 'b-task'))['benchmarks'][0]['tasks'][0]['id'], 'b-task')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            make_plan(self.args('--task', 'missing'))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            make_plan(self.args('--n-tasks', '3'))
        with self.assertRaisesRegex(ValueError, 'credentials'):
            make_plan(self.args('--api-base', 'https://user:secret@example.com/v1'))

    def test_real_harbor_configs_and_agent_imports(self):
        from harbor.agents.factory import AgentFactory
        from harbor.models.agent.name import AgentName

        plan = make_plan(self.args('--all-tasks'))
        benchmark = plan['benchmarks'][0]
        output = self.root / 'configs'
        config = build_config(plan, benchmark, output)
        self.assertEqual([t.path.name for t in config.tasks], ['a-task', 'b-task'])
        self.assertEqual(config.retry.max_retries, 0)
        self.assertEqual(config.n_attempts, 1)
        for agent in ('terminus-2', 'mini-swe-agent'):
            benchmark['agent'] = agent
            config = build_config(plan, benchmark, output)
            self.assertEqual(config.agents[0].model_name, 'openai/fixture-model')
            AgentFactory.get_agent_class(AgentName(agent))
        self.assertEqual(config.agents[0].kwargs['version'], '2.4.5')
        write_json(output / 'mini_config.json', mini_config(plan))
        agent = AgentFactory.create_agent_from_config(config.agents[0], logs_dir=output / 'logs')
        self.assertEqual(json.loads(agent._config_yaml)['agent']['step_limit'], 200)
        self.assertNotIn('API_KEY', config.model_dump_json())

    def test_incomplete_wrong_count_duplicate_and_real_zero(self):
        plan = make_plan(self.args('--all-tasks'))
        benchmark = plan['benchmarks'][0]
        output = self.root / 'result'
        for name, score in [('a-task', 1), ('b-task', 0)]:
            write_json(output / 'jobs/evaluation' / name / 'result.json', self.trial(name, score))
        trials = collect_trials(output)
        good = summarize_benchmark(self.raw(), benchmark, output, trials)
        self.assertEqual(good['score'], .5)  # a real verifier zero stays in the denominator
        self.assertIsNone(good['cost_usd'])
        for mutation in ('count', 'error', 'missing', 'duplicate', 'pending', 'nan'):
            raw, rows = copy.deepcopy(self.raw()), copy.deepcopy(trials)
            if mutation == 'count': raw['n_total_trials'] = 1
            if mutation == 'error': raw['stats']['n_errored_trials'] = 1
            if mutation == 'missing': rows.pop()
            if mutation == 'duplicate': rows[1]['task_name'] = rows[0]['task_name']
            if mutation == 'pending': raw['stats']['n_pending_trials'] = 1
            if mutation == 'nan': raw['stats']['evals']['fixture']['reward_stats']['reward'] = {'NaN': ['a', 'b']}
            with self.subTest(mutation=mutation):
                result = summarize_benchmark(raw, benchmark, output, rows)
                self.assertEqual(result['status'], 'incomplete')
                self.assertIsNone(result['score'])
        write_json(output / 'jobs/evaluation/b-task/result.json',
                   {**self.trial('b-task', 0), 'exception_info': {'exception_type': 'BuildError'}})
        self.assertIsNone(collect_trials(output)[1]['score'])
        (output / 'jobs/evaluation/b-task/result.json').write_text('{')
        self.assertEqual(collect_trials(output)[1]['status'], 'error')

    def test_preflight_and_real_http_tool_roundtrip(self):
        from harbor.models.agent.context import AgentContext
        from DataEvo.runtime.harbor_terminal_agent import TerminalAgent

        requests, commands = [], []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path == '/v1/models'
                self.reply({'data': [{'id': 'fixture-model'}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                assert self.path == '/v1/chat/completions'
                assert self.headers['Authorization'] == 'Bearer fixture-secret'
                text = '{"command":"pwd"}' if len(requests) == 1 else '{"final":"done"}'
                self.reply({'id': 'fixture', 'object': 'chat.completion', 'created': 1, 'model': 'fixture-model',
                            'choices': [{'index': 0, 'finish_reason': 'stop',
                                         'message': {'role': 'assistant', 'content': text}}],
                            'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}})

            def reply(self, value):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def log_message(self, *args):
                pass

        class FakeSandbox:
            async def exec(self, command, **kwargs):
                commands.append(command)
                return SimpleNamespace(stdout='/fixture/persistent', stderr='', return_code=0)

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            plan = make_plan(self.args('--api-base', f'http://127.0.0.1:{server.server_port}/v1'))
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-secret', 'STUDENT_API_KEY': '', 'E2B_API_KEY': ''}):
                checks = preflight(plan)
                self.assertFalse(checks['ready'])
                self.assertTrue(next(c['ok'] for c in checks['checks'] if c['name'] == 'model_endpoint'))
                self.assertFalse(next(c['ok'] for c in checks['checks'] if c['name'] == 'e2b_credentials'))
                agent = TerminalAgent(logs_dir=self.root / 'agent', model_name='fixture-model',
                                      api_base=plan['api_base'], max_turns=3)
                context = AgentContext()
                asyncio.run(agent.run('fixture instruction', FakeSandbox(), context))
            self.assertEqual(commands, ['pwd'])
            self.assertIn('/fixture/persistent', requests[1]['messages'][-1]['content'])
            self.assertEqual(context.n_input_tokens, 10)
            self.assertEqual(context.n_output_tokens, 6)
            self.assertEqual(context.metadata['termination'], 'final')
            self.assertNotIn('fixture-secret', (self.root / 'agent/trajectory.json').read_text())
            plan['benchmarks'][0]['agent'] = 'mini-swe-agent'
            checks = preflight(plan)
            self.assertFalse(next(c['ok'] for c in checks['checks'] if c['name'] == 'sandbox_model_endpoint'))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_execution_artifacts_and_exclusive_output_with_fake_job(self):
        test = self

        class FakeJob:
            def __init__(self, config):
                self.config = config

            async def run(self):
                folder = self.config.jobs_dir / self.config.job_name
                for index, task in enumerate(self.config.tasks):
                    write_json(folder / task.path.name / 'result.json',
                               test.trial(task.path.name, 1 - index))
                return SimpleNamespace(model_dump=lambda **kwargs: test.raw())

        output = self.root / 'evaluation'
        plan = make_plan(self.args('--all-tasks'))
        with patch('harbor.Job', FakeJob), patch.dict(os.environ):
            code = asyncio.run(execute(plan, output, {'ready': True}))
            self.assertEqual(code, 0)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['benchmarks'][0]['score'], .5)
            self.assertTrue((output / 'scores.csv').exists())
            self.assertIn('50.00%', (output / 'report.md').read_text())
            with self.assertRaises(FileExistsError):
                asyncio.run(execute(plan, output, {'ready': True}))
        # A setup failure in one benchmark must retain null score and allow the other to run.
        class FailingFirstJob(FakeJob):
            async def run(self):
                if self.config.tasks[0].source == 'terminal_bench_2':
                    raise RuntimeError('fixture infrastructure failure')
                return await super().run()

        second = copy.deepcopy(plan['benchmarks'][0])
        second['benchmark'] = 'swe_bench_verified'
        plan['benchmarks'].append(second)
        with patch('harbor.Job', FailingFirstJob), patch.dict(os.environ):
            self.assertEqual(asyncio.run(execute(plan, self.root / 'partial', {'ready': True})), 1)
        summaries = json.loads((self.root / 'partial/summary.json').read_text())['benchmarks']
        self.assertIsNone(summaries[0]['score'])
        self.assertEqual(summaries[0]['status'], 'error')
        self.assertEqual(summaries[1]['score'], .5)
        with patch('DataEvo.scripts.evaluate_agentic_benchmarks.preflight', return_value={'ready': False}):
            code = main(['--benchmark', 'terminal_bench_2', '--terminal-dataset', str(self.dataset),
                         '--model', 'fixture', '--output', str(self.root / 'blocked')])
            self.assertEqual(code, 2)
            self.assertFalse((self.root / 'blocked').exists())


if __name__ == '__main__':
    unittest.main()
