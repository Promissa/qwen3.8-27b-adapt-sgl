"""Reject raw/aggregate and corpus inconsistencies before auxiliary comparisons."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from compare_battery import compare, read_battery


def fixture():
    return {'status': 'passed', 'scope': 'perplexity_and_200_nonthinking_gsm8k',
        'corpus_sha256': 'a' * 64,
        'windows': [{'language': language, 'text_sha256': str(index), 'tokens': 1,
            'logprob_sum': -1., 'meta_info': {'input_token_logprobs': [[None, 0, None], [-1., index + 1, None]]}}
            for language in ('en', 'da', 'code') for index in range(40)],
        'ppl': {language: {'perplexity': math.e, 'tokens': 120 if language == 'all' else 40}
                for language in ('en', 'da', 'code', 'all')},
        'gsm8k': {'n_requested': 200, 'temperature': 0, 'thinking': False, 'max_tokens': 768,
            'corpus_sha256': 'b' * 64, 'accuracy': 1., 'truncated': 0,
            'examples': [{'question': str(index), 'gold': '1', 'prediction': '1', 'correct': True,
                'response': {'choices': [{'message': {'content': 'Final answer: 1'}, 'finish_reason': 'stop'}],
                             'usage': {'completion_tokens': 10}}} for index in range(200)]}}


class BatteryEvidenceTests(unittest.TestCase):
    def test_complete_matched_control_passes_without_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'battery.json'; path.write_text(json.dumps(fixture()))
            result = compare(path, path)
            self.assertTrue(result['matched_evidence_valid'])
            self.assertFalse(result['full_plan_promotion_verified'])
            self.assertEqual(result['greedy_accuracy_delta_percentage_points'], 0)

    def test_partial_windows_and_false_prediction_are_rejected(self):
        for kind in ('partial', 'prediction', 'gold'):
            data = fixture()
            if kind == 'partial':
                data['windows'].pop()
            elif kind == 'prediction':
                data['gsm8k']['examples'][0]['response']['choices'][0]['message']['content'] = 'Final answer: 2'
            else:
                data['gsm8k']['examples'][0]['gold'] = 'nan'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'battery.json'; path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): read_battery(path)

    def test_hidden_truncation_and_wrong_logprob_sum_are_rejected(self):
        for kind in ('truncation', 'logprob'):
            data = fixture()
            if kind == 'truncation':
                response = data['gsm8k']['examples'][0]['response']
                response['choices'][0]['finish_reason'] = 'length'
                response['usage']['completion_tokens'] = 768
            else:
                data['windows'][0]['logprob_sum'] = -2.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'battery.json'; path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): read_battery(path)

    def test_different_corpus_and_tokenization_are_rejected(self):
        for kind in ('corpus', 'tokens'):
            left = fixture(); right = copy.deepcopy(left)
            if kind == 'corpus': right['corpus_sha256'] = 'c' * 64
            else: right['windows'][0]['meta_info']['input_token_logprobs'][1][1] = 900
            with tempfile.TemporaryDirectory() as directory:
                a, b = Path(directory) / 'a.json', Path(directory) / 'b.json'
                a.write_text(json.dumps(left)); b.write_text(json.dumps(right))
                with self.assertRaises(ValueError): compare(a, b)


if __name__ == '__main__':
    unittest.main()
