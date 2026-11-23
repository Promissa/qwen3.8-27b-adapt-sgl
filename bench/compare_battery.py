#!/usr/bin/env python3
"""Recompute complete auxiliary evidence and compare matched baseline/candidate inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from common import now, write_json


def read_battery(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get('status') != 'passed' or data.get('scope') != 'perplexity_and_200_nonthinking_gsm8k':
        raise ValueError('Require a complete auxiliary battery, not PPL-only or partial evidence')
    windows = data['windows']
    if Counter(row['language'] for row in windows) != {'en': 40, 'da': 40, 'code': 40}:
        raise ValueError('Require all120 windows,40 per language')
    if len({(row['language'], row['text_sha256']) for row in windows}) != 120:
        raise ValueError('Duplicate PPL windows')
    stats = defaultdict(lambda: [0., 0])
    for row in windows:
        logs = [entry[0] for entry in row['meta_info']['input_token_logprobs'] if entry[0] is not None]
        if not logs or any(not math.isfinite(value) or value > 0 for value in logs):
            raise ValueError('Invalid raw input-token log probabilities')
        if len(logs) != row['tokens'] or sum(logs) != row['logprob_sum']:
            raise ValueError('PPL window sums/counts disagree with raw probabilities')
        stats[row['language']][0] += sum(logs)
        stats[row['language']][1] += len(logs)
    stats['all'] = [sum(value[0] for value in stats.values()), sum(value[1] for value in stats.values())]
    ppl = {}
    for language, (total, count) in stats.items():
        value = math.exp(-total / count)
        recorded = data['ppl'][language]
        if recorded['tokens'] != count or not math.isclose(recorded['perplexity'], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError('PPL aggregate disagrees with raw probabilities')
        ppl[language] = {'perplexity': value, 'tokens': count}
    gsm = data['gsm8k']
    protocol = {key: gsm[key] for key in ('n_requested', 'temperature', 'thinking', 'max_tokens', 'corpus_sha256')}
    if (protocol['n_requested'], protocol['temperature'], protocol['thinking'], protocol['max_tokens']) != (200, 0, False, 768):
        raise ValueError('Auxiliary greedy protocol differs from the control')
    examples = gsm['examples']
    if len(examples) != 200 or len({row['question'] for row in examples}) != 200:
        raise ValueError('Require200 unique greedy examples')
    correct, truncated = 0, 0
    for row in examples:
        response = row['response']
        choices = response['choices']
        if len(choices) != 1 or choices[0]['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Unsuccessful greedy response')
        generated = response['usage']['completion_tokens']
        if not isinstance(generated, int) or not 0 < generated <= 768:
            raise ValueError('Invalid actual greedy output budget')
        if choices[0]['finish_reason'] == 'length':
            if generated != 768:
                raise ValueError('Silently clamped greedy output')
            truncated += 1
        text = choices[0]['message'].get('content') or ''
        answer = re.search(r'Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)', text)
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace('$', ''))
        predicted = answer.group(1).replace(',', '') if answer else numbers[-1].replace(',', '') if numbers else None
        if predicted != row['prediction']:
            raise ValueError('Stored prediction disagrees with actual response text')
        gold = float(row['gold'])
        if not math.isfinite(gold):
            raise ValueError('Nonfinite gold answer')
        try:
            is_correct = abs(float(predicted) - gold) < 1e-6
        except (TypeError, ValueError):
            is_correct = False
        if not isinstance(row['correct'], bool) or row['correct'] != is_correct:
            raise ValueError('Stored greedy score disagrees with response answer')
        correct += is_correct
    if gsm['accuracy'] != correct / 200 or gsm['truncated'] != truncated:
        raise ValueError('Greedy aggregate or truncation count disagrees with raw responses')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'ppl': ppl, 'greedy_correct': correct, 'greedy_accuracy_percent': correct / 2,
            'greedy_truncated': truncated, 'protocol': protocol,
            'corpus_sha256': data['corpus_sha256'], 'windows': windows, 'examples': examples}


def compare(baseline_path, candidate_path):
    baseline, candidate = read_battery(baseline_path), read_battery(candidate_path)
    if baseline['corpus_sha256'] != candidate['corpus_sha256'] or baseline['protocol'] != candidate['protocol']:
        raise ValueError('Different auxiliary corpus or generation protocol')
    for left, right in zip(baseline['windows'], candidate['windows']):
        if (left['language'], left['text_sha256']) != (right['language'], right['text_sha256']):
            raise ValueError('Different PPL window content/order')
        if [entry[1] for entry in left['meta_info']['input_token_logprobs']] != [entry[1] for entry in right['meta_info']['input_token_logprobs']]:
            raise ValueError('Different tokenized PPL inputs')
    if [(row['question'], row['gold']) for row in baseline['examples']] != [(row['question'], row['gold']) for row in candidate['examples']]:
        raise ValueError('Different greedy questions or gold answers')
    for report in (baseline, candidate):
        del report['windows'], report['examples']
    return {'at': now(), 'matched_evidence_valid': True, 'baseline': baseline, 'candidate': candidate,
            'perplexity_relative_change_percent': {language: 100 * (candidate['ppl'][language]['perplexity'] / baseline['ppl'][language]['perplexity'] - 1) for language in baseline['ppl']},
            'greedy_accuracy_delta_percentage_points': candidate['greedy_accuracy_percent'] - baseline['greedy_accuracy_percent'],
            'greedy_truncation_delta': candidate['greedy_truncated'] - baseline['greedy_truncated'],
            'full_plan_promotion_verified': False,
            'scope': 'Complete matched auxiliary evidence and measured deltas only; no invented auxiliary acceptance threshold or final recipe promotion'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new evidence file')
    write_json(args.output, compare(args.baseline, args.candidate))
