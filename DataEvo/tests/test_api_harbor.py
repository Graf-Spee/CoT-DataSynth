from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from DataEvo.runtime.api import complete
from DataEvo.runtime.common import Store
from DataEvo.runtime.harbor_job import summarize


class AccountingTest(unittest.TestCase):
    def test_real_http_client_reserves_budget_and_records_usage(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['model'] == 'test-teacher'
                assert self.headers['Authorization'] == 'Bearer never-store-this-key'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'message': {'content': '#### 2'}}],
                    'usage': {'total_tokens': 12, 'prompt_tokens': 8, 'completion_tokens': 4}}).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'runner').mkdir()
                context = {'session': tmp, 'attempt_id': 'a0000', 'budget': {'max_api_tokens': 1000},
                    'teacher_api': {'base_url': f'http://127.0.0.1:{server.server_port}/v1',
                        'api_key_env': 'TEST_SYNTHESIS_KEY', 'model': 'test-teacher',
                        'input_usd_per_million': 1, 'output_usd_per_million': 2}}
                with patch.dict('os.environ', {'TEST_SYNTHESIS_KEY': 'never-store-this-key'}):
                    result = complete(context, [{'role': 'user', 'content': '1+1?'}], max_tokens=20)
                    self.assertEqual(result['choices'][0]['message']['content'], '#### 2')
                    usage = Store(root).costs()[0]
                    self.assertEqual(usage['tokens'], 12)
                    self.assertAlmostEqual(usage['known_cost_usd'], .000016)
                    context['budget']['max_api_tokens'] = 12
                    with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                        complete(context, [{'role': 'user', 'content': '1+1?'}], max_tokens=20)
                self.assertNotIn(b'never-store-this-key', (root / 'runner/session.sqlite').read_bytes())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_harbor_denominator_and_incomplete_failure(self):
        raw = {'n_total_trials': 4, 'stats': {'n_completed_trials': 4, 'n_errored_trials': 0,
                'evals': {'agent__dataset': {'reward_stats': {'reward': {'0.0': ['a', 'b'], '1.0': ['c', 'd']}}}}}}
        self.assertEqual(summarize(raw, Path('/tmp'))['score'], .5)
        custom = json.loads(json.dumps(raw).replace('"reward":', '"success":'))
        self.assertEqual(summarize(custom, Path('/tmp'), 'success')['score'], .5)
        invalid = json.loads(json.dumps(raw).replace('"0.0":', '"-1.0":'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            summarize(invalid, Path('/tmp'))
        raw['stats']['n_errored_trials'] = 1
        with self.assertRaisesRegex(ValueError, 'incomplete Harbor'):
            summarize(raw, Path('/tmp'))
        raw['stats']['n_errored_trials'] = 0
        raw['n_total_trials'] = 5
        with self.assertRaisesRegex(ValueError, 'incomplete Harbor'):
            summarize(raw, Path('/tmp'))


if __name__ == '__main__':
    unittest.main()
