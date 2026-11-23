"""Retained responses need exact inputs, full budgets and official rescoring."""
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from resume_quality import (sample_function, source_protocol, validate_rows,
                            validate_serving_configuration, validate_experiment_environment)
from sgl_eval.types import Example, Sample


class QualityResumeTests(unittest.TestCase):
    def examples(self):
        return [Example(id='gsm8k-0', inputs={'problem': 'What is 6 times 7?'}, target='42'),
                Example(id='gsm8k-1', inputs={'problem': 'What is 3 plus 4?'}, target='7')]

    def row(self):
        return {'id': 'gsm8k-0', 'problem': self.examples()[0].inputs['problem'],
                'expected_answer': '42', 'generation': 'The answer is \\boxed{42}.',
                'num_generated_tokens': 12, 'finish_reason': 'stop',
                'symbolic_correct': True, 'predicted_answer': '42'}

    def test_duplicate_mismatched_input_and_silently_shortened_cap_rejected(self):
        row = self.row()
        self.assertEqual(len(validate_rows([row], self.examples(), 16384)), 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows([row, row], self.examples(), 16384)
        with self.assertRaisesRegex(ValueError, 'question'):
            validate_rows([dict(row, problem='Different question')], self.examples(), 16384)
        with self.assertRaisesRegex(ValueError, 'cap'):
            validate_rows([dict(row, finish_reason='length')], self.examples(), 16384)

    def test_cached_generation_is_regraded_by_official_runner_and_only_missing_id_calls_transport(self):
        from sgl_eval.evals._math import MathEvaluator, make_score_one_fn
        from sgl_eval.predictions import PredictionsWriter
        from sgl_eval.runner import run_examples

        examples = self.examples()
        retained = validate_rows([self.row()], examples, 16384)
        called = []

        def fresh(example, repeat):
            called.append((example.id, repeat))
            return Sample(text='The answer is \\boxed{7}.', completion_tokens=12, finish_reason='stop')

        with tempfile.TemporaryDirectory() as tmp:
            with PredictionsWriter(Path(tmp), 1) as writer:
                result = run_examples(name='gsm8k', examples=examples,
                    sample_fn=sample_function(retained, fresh, [10, 10]),
                    score_one_fn=make_score_one_fn(MathEvaluator(config={})),
                    num_threads=1, n_repeats=1, progress=False, on_sample_scored=writer)
            rows = [json.loads(line) for line in (Path(tmp) / 'output-rs0.jsonl').read_text().splitlines()]
        self.assertEqual(called, [('gsm8k-1', 0)])
        self.assertEqual(result.num_examples, 2)
        self.assertFalse(result.partial)
        self.assertEqual(result.aggregate['score'], 1)
        self.assertEqual(rows[0]['generation'], self.row()['generation'])
        self.assertTrue(all(row['symbolic_correct'] for row in rows))

    def test_original_protocol_must_be_unique_and_seeded(self):
        protocol = {'max_tokens': 16384, 'temperature': 1., 'top_p': .95, 'min_p': 0.,
                    'seed': 1701, 'chat_template_kwargs': {'enable_thinking': True}, 'top_k': 20}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'validation-03.log'
            path.write_text('sgl-eval gsm8k generation: ' + repr(protocol) + '\n')
            self.assertEqual(source_protocol(root), protocol)
            path.write_text('sgl-eval gsm8k generation: ' + repr(dict(protocol, seed=None)) + '\n')
            with self.assertRaisesRegex(ValueError, 'unseeded'):
                source_protocol(root)

    def test_resume_rejects_precision_backend_graph_and_speculation_changes(self):
        original = {'model_path': '/weights/target', 'kv_cache_dtype': 'nvfp4',
            'mamba_ssm_dtype': 'bfloat16', 'attention_backend': 'flashinfer',
            'prefill_attention_backend': 'flashinfer', 'decode_attention_backend': 'trtllm_mha',
            'cuda_graph_config': {'decode': {'backend': 'full'}},
            'enable_deterministic_inference': False, 'speculative_algorithm': None,
            'random_seed': 1, 'startup_time': {'load_weight': 10}, 'internal_states': [1]}
        restarted = copy.deepcopy(original)
        restarted.update(random_seed=2, startup_time={'load_weight': 20}, internal_states=[2])
        identity = validate_serving_configuration(original, restarted, 1701)
        self.assertEqual(identity['compared_fields'], 9)
        for key, value in [('kv_cache_dtype', 'fp8_e4m3'), ('mamba_ssm_dtype', 'float32'),
                           ('decode_attention_backend', 'triton'), ('speculative_algorithm', 'NGRAM'),
                           ('enable_deterministic_inference', True),
                           ('cuda_graph_config', {'decode': {'backend': 'disabled'}})]:
            with self.subTest(key=key):
                changed = copy.deepcopy(restarted)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_serving_configuration(original, changed, 1701)
        with self.assertRaisesRegex(ValueError, 'configuration evidence'):
            validate_serving_configuration(None, restarted, 1701)
        with self.assertRaisesRegex(ValueError, 'integer per-request seed'):
            validate_serving_configuration(original, restarted, None)

    def test_resume_environment_requires_identical_engine_switches(self):
        original = {'SGLANG_OPT_GDN_VERIFY_CONV_ORDER': '1', 'MAX_JOBS': '2'}
        self.assertEqual(validate_experiment_environment(original, dict(original, PATH='/different'))['compared_fields'], 2)
        for changed in ({'MAX_JOBS': '2'}, dict(original, SGLANG_OPT_GDN_VERIFY_MATCH_PACKED='1')):
            with self.assertRaisesRegex(ValueError, 'experimental environment'):
                validate_experiment_environment(original, changed)
        with self.assertRaisesRegex(ValueError, 'Missing original'):
            validate_experiment_environment(None, original)


if __name__ == '__main__':
    unittest.main()
