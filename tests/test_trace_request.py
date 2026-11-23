"""Reject a complete-looking trace from a different actual prompt."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import capture_forward_trace as capture


class TraceRequestTests(unittest.TestCase):
    def test_real_prompt_ids_must_match_even_when_positions_and_emitted_root_match(self):
        for wrong_prompt in (False, True):
            with self.subTest(wrong_prompt=wrong_prompt), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                worker = folder / 'trace/worker-fixture'
                worker.mkdir(parents=True)
                manifest = worker / 'trace.json'
                manifest.write_text(json.dumps({'status': 'unarmed', 'frames': [],
                    'source_sha256': hashlib.sha256((ROOT / 'bench/forward_trace.py').read_bytes()).hexdigest()}))
                arm = folder / 'armed'
                output = folder / 'capture.json'
                expected_prompt = json.loads((ROOT / 'bench/prompts_real.jsonl').read_text().splitlines()[0])['prompt']
                prompt_ids = [10, 11, 12]
                routes = []

                def api(base, route, payload=None):
                    routes.append(route)
                    if route == '/get_server_info':
                        return {'model_path': 'owned-model'}
                    if route == '/tokenize':
                        self.assertEqual(payload, {'prompt': expected_prompt, 'add_special_tokens': True})
                        self.assertFalse(arm.exists())
                        return {'tokens': prompt_ids, 'count': 3, 'max_model_len': 24576}
                    if route == '/flush_cache':
                        self.assertFalse(arm.exists())
                        return 'Success'
                    if route == '/generate':
                        self.assertTrue(arm.exists())
                        self.assertEqual(payload['text'], expected_prompt)
                        actual = [10, 99, 12] if wrong_prompt else prompt_ids
                        manifest.write_text(json.dumps({'status': 'complete', 'frames': [
                            {'input_ids': actual, 'positions': [0, 1, 2]},
                            {'input_ids': [7], 'positions': [3]}]}))
                        return {'output_ids': list(range(7, 15)), 'meta_info': {'completion_tokens': 8}}
                    self.fail('Unexpected endpoint: ' + route)

                argv = ['capture_forward_trace.py', '--trace-root', str(folder / 'trace'),
                        '--arm-file', str(arm), '--output', str(output)]
                with patch.object(sys, 'argv', argv), patch.object(capture, 'api', api):
                    rc = capture.main()
                result = json.loads(output.read_text())
                self.assertEqual(rc, 1 if wrong_prompt else 0)
                self.assertEqual(result['status'], 'failed' if wrong_prompt else 'passed')
                self.assertEqual(result['prompt_tokenization']['tokens'], prompt_ids)
                self.assertFalse(arm.exists())
                self.assertLess(routes.index('/tokenize'), routes.index('/generate'))
                if wrong_prompt:
                    self.assertIn('requested prompt token IDs', result['error'])


if __name__ == '__main__':
    unittest.main()
