#!/usr/bin/env python3
"""Collect completed budgeted evidence without converting screening to full gates."""
import argparse
import hashlib
import json
from pathlib import Path

from common import ROOT, now, write_json
from summarize_serving import validate
from token_compare import compare
from compare_quality import read_run

EXPECTED = ['baseline', 'balanced', 'max-context', 'qat', 'state-fp32',
            'gemm-cudnn', 'gdn-decode-flashinfer', 'gdn-prefill-flashinfer',
            'effort-low', 'baseline-after']


def collect():
    folder = ROOT / 'results/budget-3h-20261112'
    queue = json.loads((folder / 'runs.json').read_text())
    baseline_dir = json.loads((ROOT / 'results/quality/baseline-full-16k-gsm8k-validated.json').read_text())['directory']
    baseline_quality = read_run(baseline_dir)
    result = {'at': now(), 'queue_status': queue['status'], 'trials': [],
        'missing_trials': [], 'full_original_plan_verified': False,
        'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Actual deadline-bounded reduced evidence; no inferred full quality, repeat-noise, physical-residency or two-hour final-combination gate'}
    found = {r['tag'] for r in queue['trials']}
    result['missing_trials'] = [f'budget-{name}-r1' for name in EXPECTED
                                if f'budget-{name}-r1' not in found]
    for entry in queue['trials']:
        row = dict(entry)
        tag = row['tag']
        if not row.get('probe'):
            result['trials'].append(row)
            continue
        probe_path = ROOT / row['probe']
        probe = json.loads(probe_path.read_text())
        row['probe_sha256'] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        row['ready_s'] = probe.get('ready_s')
        row['peak_device_used_mib'] = probe.get('peak_device_used_mib')
        row['launch_command'] = probe.get('command')
        row['capacity_passed'] = probe.get('capacity_gate_passed') is True
        row['runtime_provenance'] = probe.get('runtime_provenance')
        row['target_revision'] = probe.get('target_revision')
        row['target_conversion'] = probe.get('target_conversion')
        row['experiment_environment'] = probe.get('experiment_environment')
        provenance = probe.get('runtime_provenance', {})
        if provenance.get('engine_patch'):
            patch = ROOT / provenance['engine_patch']
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            if digest != provenance['engine_diff_sha256']:
                raise ValueError('Changed launch-specific engine patch: ' + tag)
            row['engine_patch_hash_verified'] = True
        summary = ROOT / f'results/bench/{tag}/screening-summary.json'
        if summary.exists():
            data = json.loads(summary.read_text())
            if data['status'] == 'completed_screening':
                if {g['input_tokens'] for g in data['groups']} != {1024, 8192} or len(data['groups']) != 2:
                    raise ValueError('Incomplete two-load screening benchmark')
                for group in data['groups']:
                    path = ROOT / group['raw']
                    if hashlib.sha256(path.read_bytes()).hexdigest() != group['raw_sha256']:
                        raise ValueError('Changed raw serving evidence: ' + tag)
                    raw, = path.read_text().splitlines()
                    actual = json.loads(raw)
                    validate(actual, group['input_tokens'])
                    if actual['completed'] != 1 or any(group['metrics'][key] != actual[key] for key in group['metrics']):
                        raise ValueError('Serving summary differs from raw single-request metrics')
                row['serving'] = data
        for suffix, key in [('screening', 'quality'), ('auxiliary-screening', 'auxiliary')]:
            path = ROOT / f'results/quality/{tag}-{suffix}.json'
            if path.exists():
                data = json.loads(path.read_text())
                if data.get('full_plan_promotion_verified') is not False:
                    raise ValueError('Reduced evidence must not promote a full recipe')
                if key == 'quality':
                    directory = Path(data['run_directory'])
                    for name, digest in data['evidence_sha256'].items():
                        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                            raise ValueError('Changed raw screening quality')
                    if data['n'] != len(data['official_rescore']):
                        raise ValueError('Incomplete official subset rescore')
                    if sum(r['symbolic_correct'] for r in data['official_rescore']) != data['correct']:
                        raise ValueError('Official subset rescore total differs')
                    predictions = [json.loads(line) for line in (directory / 'output-rs0.jsonl').read_text().splitlines()]
                    data['generated_tokens'] = sum(r['num_generated_tokens'] for r in predictions)
                    data['reasoning_tokens'] = sum(r['num_reasoning_tokens'] for r in predictions)
                    data['generation_seconds'] = sum(r['generation_end_time'] - r['generation_start_time'] for r in predictions)
                    data['baseline_subset_generated_tokens'] = sum(baseline_quality['rows'][r['id']]['num_generated_tokens'] for r in predictions)
                else:
                    raw_path = ROOT / f'results/quality/{tag}/battery.json'
                    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != data['raw_sha256']:
                        raise ValueError('Changed auxiliary raw evidence')
                row[key] = data
        validation = ROOT / f'results/validation/{tag}'
        retrieval = validation / 'retrieval.json'
        if retrieval.exists():
            data = json.loads(retrieval.read_text())
            row['retrieval'] = {'status': data['status'], 'reported_capacity': data.get('reported_capacity'),
                'resolved_input_tokens': data.get('resolved_input_tokens'),
                'prefix_hit_ttft_gate': data.get('prefix_hit_ttft_gate'),
                'checks': [{'depth': r['depth'], 'retrieval_passed': r['retrieval_passed'],
                    'input_tokens': r['input_tokens'], 'prefix_hit_observed': r['prefix_hit_observed'],
                    'warm_ttft_s': r['warm']['ttft_s']} for r in data['checks']]}
        tokens = validation / 'tokens.json'
        if tokens.exists():
            data = json.loads(tokens.read_text())
            control = ('results/validation/embedding-full-resumed-20261112/tokens.json'
                if row['model'] == 'target-embedding-nvfp4' else 'results/validation/baseline-088/tokens.json')
            reference = json.loads((ROOT / control).read_text())
            row['token_capture_complete'] = data['complete']
            row['strict_tokens_vs_retained_control'] = compare(reference, data) if data['complete'] else False
            row['token_control'] = control
        soak = ROOT / f'results/soak/{tag}/soak.json'
        if soak.exists():
            data = json.loads(soak.read_text())
            row['short_soak'] = {k: data.get(k) for k in ['status', 'requested_seconds',
                'elapsed_s', 'full_plan_soak_gate', 'runtime_stability_7200s_gate', 'telemetry_gate', 'error']}
        harness = ROOT / f'results/harness/{tag}/harness.json'
        if harness.exists():
            data = json.loads(harness.read_text())
            row['harness'] = {k: data.get(k) for k in ['status', 'version', 'elapsed_s', 'error']}
        result['trials'].append(row)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing audits; use a fresh output')
    write_json(args.output, collect())
