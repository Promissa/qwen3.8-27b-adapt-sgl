#!/usr/bin/env python3
"""Explicit short screening; never grants the full quality/performance gates."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from common import ROOT, now, write_json
from compare_quality import read_run
from compare_battery import read_battery
from summarize_serving import METRICS, validate


def short_bench(tag):
    folder = ROOT / 'results/bench' / tag
    folder.mkdir(parents=True, exist_ok=True)
    report = {'at': now(), 'status': 'running', 'groups': [],
              'repetitions': 1, 'requests_per_load': 1,
              'scope': 'One request at each1k/8k input load,1024 output; no repeat-noise estimate or final speed qualification'}
    destination = folder / 'screening-summary.json'
    if destination.exists():
        raise FileExistsError(destination)
    try:
        model = os.environ.get('MODEL_PATH', str(ROOT / 'weights/target'))
        for isl in (1024, 8192):
            raw = folder / f'isl{isl}-rep1.jsonl'
            if raw.exists():
                raise FileExistsError(raw)
            command = [sys.executable, '-m', 'sglang.benchmark.serving',
                '--backend', 'sglang-oai', '--host', '127.0.0.1', '--port', '30000',
                '--model', model, '--served-model-name', 'qwen3.8-27b', '--tokenizer', model,
                '--dataset-name', 'random', '--random-input-len', str(isl),
                '--random-output-len', '1024', '--random-range-ratio', '1',
                '--num-prompts', '1', '--max-concurrency', '1', '--request-rate', 'inf',
                '--flush-cache', '--output-details', '--output-file', str(raw)]
            with raw.with_suffix('.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=240, check=True)
            lines = raw.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError('Expected one complete raw benchmark row')
            row = json.loads(lines[0])
            validate(row, isl)
            if row['completed'] != 1:
                raise ValueError('Screening requires exactly one request per load')
            report['groups'].append({'input_tokens': isl, 'output_tokens': 1024,
                'metrics': {key: row[key] for key in METRICS},
                'raw': str(raw.relative_to(ROOT)), 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()})
            write_json(destination, report)
        report['status'] = 'completed_screening'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination, report)


def read_subset(directory):
    metrics_path, predictions_path = directory / 'metrics.json', directory / 'output-rs0.jsonl'
    metrics = json.loads(metrics_path.read_text())
    n = metrics.get('num_examples')
    if metrics.get('name') != 'gsm8k' or type(n) is not int or not 1 <= n < 1319 or metrics.get('n_repeats') != 1:
        raise ValueError('Require an explicitly reduced, completed one-repeat GSM8K run')
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    if len(rows) != n or {r['id'] for r in rows} != {f'gsm8k-{i}' for i in range(n)}:
        raise ValueError('Missing, duplicate or unexpected subset IDs')
    cap = metrics['generation']['max_tokens']
    for row in rows:
        tokens = row.get('num_generated_tokens')
        if (type(row.get('symbolic_correct')) is not bool or row.get('finish_reason') not in ('stop', 'length')
                or type(tokens) is not int or not 0 < tokens <= cap
                or (row['finish_reason'] == 'length' and tokens != cap)):
            raise ValueError('Incomplete response or changed actual output cap')
    correct = sum(r['symbolic_correct'] for r in rows)
    if (not math.isclose(metrics['aggregate']['score'], correct / n, rel_tol=0, abs_tol=1e-12)
            or metrics['aggregate'].get('error_rate', 0) != 0):
        raise ValueError('Aggregate disagrees with complete successful raw rows')
    return metrics, rows


def quality_report(tag, allow_effort_change=False):
    runs = list((ROOT / 'results/gsm8k' / tag).glob('sgl_eval_*'))
    if len(runs) != 1:
        raise ValueError('Require one completed screening run')
    directory = runs[0]
    metrics, rows = read_subset(directory)
    baseline_dir = json.loads((ROOT / 'results/quality/baseline-full-16k-gsm8k-validated.json').read_text())['directory']
    baseline = read_run(baseline_dir)
    matched = metrics['generation'] == baseline['protocol']
    if not matched and not allow_effort_change:
        raise ValueError('Generation protocol differs from the full baseline')
    if metrics['generation']['max_tokens'] != baseline['protocol']['max_tokens']:
        raise ValueError('The output cap must remain matched')
    from sgl_eval.evals._math import MathEvaluator
    from sgl_eval.evals._prompts import resolve_prompt
    import yaml
    config = yaml.safe_load(resolve_prompt('math').read_text()).get('evaluator_config', {})
    scorer = MathEvaluator(config=config or {})
    if scorer.eval_config.use_predicted_answer_key:
        raise ValueError('Official rescore must extract from generation text')
    lost, gained, rescored = [], [], []
    for row in rows:
        control = baseline['rows'][row['id']]
        if (row['problem'], row['expected_answer']) != (control['problem'], control['expected_answer']):
            raise ValueError('Subset question/gold differs from the baseline')
        grade = asyncio.run(scorer.eval_single({'generation': row['generation'], 'expected_answer': row['expected_answer']}))
        if bool(grade['symbolic_correct']) != row['symbolic_correct'] or grade['predicted_answer'] != row['predicted_answer']:
            raise ValueError('Independent official extraction/label disagreement')
        rescored.append({'id': row['id'], 'symbolic_correct': bool(grade['symbolic_correct'])})
        if control['symbolic_correct'] and not row['symbolic_correct']:
            lost.append(row['id'])
        if row['symbolic_correct'] and not control['symbolic_correct']:
            gained.append(row['id'])
    n = len(rows)
    report = {'at': now(), 'status': 'completed_screening_audit', 'n': n,
        'correct': sum(r['symbolic_correct'] for r in rows),
        'baseline_subset_correct': sum(baseline['rows'][r['id']]['symbolic_correct'] for r in rows),
        'protocol_matched': matched, 'protocol': metrics['generation'],
        'lost_correct_ids': lost, 'gained_correct_ids': gained, 'official_rescore': rescored,
        'sample_resolution_percentage_points': 100 / n, 'full_quality_gate_evaluated': False,
        'full_plan_promotion_verified': False, 'run_directory': str(directory),
        'evidence_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (directory / 'metrics.json', directory / 'output-rs0.jsonl')},
        'scope': 'Reduced screening with complete raw rows and fresh official rescore; cannot establish the0.5pp full-GSM8K gate'}
    destination = ROOT / f'results/quality/{tag}-screening.json'
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, report)


def auxiliary_report(tag):
    baseline = read_battery(ROOT / 'results/quality/baseline-logprob-chunk128/battery.json')
    path = ROOT / f'results/quality/{tag}/battery.json'
    data = json.loads(path.read_text())
    if data.get('status') != 'passed' or data.get('scope') != 'reduced_screening_battery':
        raise ValueError('Require a completed explicitly reduced auxiliary battery')
    k, n = data['requested_windows_per_domain'], data['requested_greedy_examples']
    selected, counts = [], {}
    for row in baseline['windows']:
        language = row['language']
        if counts.get(language, 0) < k:
            selected.append(row)
            counts[language] = counts.get(language, 0) + 1
    windows = data['windows']
    if len(windows) != 3 * k or len(selected) != len(windows):
        raise ValueError('Incomplete auxiliary window subset')
    totals = [0., 0]
    control_totals = [0., 0]
    for row, control in zip(windows, selected):
        if (row['language'], row['text_sha256']) != (control['language'], control['text_sha256']):
            raise ValueError('Different auxiliary window selection')
        logs = [x[0] for x in row['meta_info']['input_token_logprobs'] if x[0] is not None]
        if (not logs or any(not math.isfinite(x) or x > 0 for x in logs)
                or len(logs) != row['tokens'] or sum(logs) != row['logprob_sum']):
            raise ValueError('Auxiliary raw probabilities disagree with sums/counts')
        if ([x[1] for x in row['meta_info']['input_token_logprobs']]
                != [x[1] for x in control['meta_info']['input_token_logprobs']]):
            raise ValueError('Auxiliary tokenized inputs differ')
        totals[0] += sum(logs)
        totals[1] += len(logs)
        control_totals[0] += control['logprob_sum']
        control_totals[1] += control['tokens']
    examples = data['gsm8k']['examples']
    if len(examples) != n or not 1 <= n <= 200:
        raise ValueError('Incomplete reduced greedy examples')
    correct = 0
    for row, control in zip(examples, baseline['examples'][:n]):
        if (row['question'], row['gold']) != (control['question'], control['gold']):
            raise ValueError('Greedy question/gold differs')
        choice = row['response']['choices'][0]
        generated = row['response']['usage']['completion_tokens']
        if type(generated) is not int or not 0 < generated <= 768 or (choice['finish_reason'] == 'length' and generated != 768):
            raise ValueError('Greedy output budget differs from the retained control')
        text = choice['message'].get('content') or ''
        match = re.search(r'Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)', text)
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace('$', ''))
        prediction = match.group(1).replace(',', '') if match else numbers[-1].replace(',', '') if numbers else None
        try:
            label = abs(float(prediction) - float(row['gold'])) < 1e-6
        except (TypeError, ValueError):
            label = False
        if prediction != row['prediction'] or label != row['correct'] or choice['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Greedy raw answer/label/termination differs')
        correct += label
    ppl = math.exp(-totals[0] / totals[1])
    if not math.isclose(ppl, data['ppl']['all']['perplexity'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Reduced PPL aggregate differs from raw windows')
    if data['gsm8k']['accuracy'] != correct / n:
        raise ValueError('Reduced greedy aggregate differs from raw answers')
    destination = ROOT / f'results/quality/{tag}-auxiliary-screening.json'
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, {'at': now(), 'status': 'completed_screening_audit',
        'windows_per_domain': k, 'greedy_examples': n, 'greedy_correct': correct,
        'baseline_subset_greedy_correct': sum(r['correct'] for r in baseline['examples'][:n]),
        'ppl': ppl, 'baseline_subset_ppl': math.exp(-control_totals[0] / control_totals[1]),
        'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'full_plan_promotion_verified': False, 'scope': 'Matched reduced auxiliary inputs, recomputed raw probabilities/answers; not the full120-window/200-example battery'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bench', 'quality-report', 'aux-report'])
    parser.add_argument('tag')
    parser.add_argument('--separate-effort', action='store_true')
    args = parser.parse_args()
    if args.mode == 'bench':
        short_bench(args.tag)
    elif args.mode == 'quality-report':
        quality_report(args.tag, args.separate_effort)
    else:
        auxiliary_report(args.tag)
