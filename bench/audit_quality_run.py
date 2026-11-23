#!/usr/bin/env python3
"""Audit completed quality evidence without generation or recipe promotion."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from common import ROOT, now, write_json
from compare_battery import compare as compare_battery
from compare_quality import compare, read_run


def validate_rescore(report, run):
    if (report.get('status') != 'passed' or report.get('checked') != 1319
            or report.get('n') != 1319 or report.get('correct') != run['correct']
            or report.get('run_directory') != run['directory']
            or report.get('original_evidence_sha256') != run['evidence_sha256']
            or report.get('score_mismatch_ids') != [] or report.get('extraction_mismatch_ids') != []):
        raise ValueError('Official rescore does not bind complete current evidence')
    rows = report.get('rows', [])
    identifiers = [row['id'] for row in rows]
    if len(identifiers) != 1319 or set(identifiers) != set(run['rows']):
        raise ValueError('Official rescore row coverage differs')
    for row in rows:
        old = run['rows'][row['id']]
        if any(row[key] != old[key] for key in ('expected_answer', 'predicted_answer', 'symbolic_correct')):
            raise ValueError('Official rescore row disagrees: ' + row['id'])


def audit(tag, output, separate_low_effort=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag):
        raise ValueError('Require a plain evaluation tag')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'at': now(), 'status': 'running', 'tag': tag,
        'validation_python': sys.version, 'validation_executable': sys.executable,
        'separate_low_effort': separate_low_effort, 'full_plan_promotion_verified': False,
        'source_sha256': {name: hashlib.sha256((ROOT / 'bench' / name).read_bytes()).hexdigest()
            for name in ('audit_quality_run.py', 'rescore_quality.py', 'compare_quality.py', 'compare_battery.py')},
        'scope': 'Full raw GSM8K, fresh official re-extraction/rescoring, and matched auxiliary evidence. A completed audit can contain a failed GSM8K quality gate. No generation, performance, capacity or final-recipe promotion.'}
    destination = output / 'audit.json'
    write_json(destination, report)
    try:
        probe = ROOT / f'results/probes/{tag}/probe.json'
        info = json.loads(probe.read_text())
        report.update(probe=str(probe), probe_status=info['status'],
                      probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
        if info['status'] == 'running':
            raise ValueError('Evaluation probe is still running')
        metrics = list((ROOT / f'results/gsm8k/{tag}').glob('*/metrics.json'))
        if len(metrics) != 1:
            raise ValueError('Require exactly one completed quality run')
        run = read_run(metrics[0].parent)
        if run['protocol']['max_tokens'] != 16384:
            raise ValueError('Require the unchanged16384-token quality budget')
        baseline_record = ROOT / 'results/quality/baseline-full-16k-gsm8k-validated.json'
        baseline_dir = Path(json.loads(baseline_record.read_text())['directory'])
        baseline = read_run(baseline_dir)
        report['candidate'] = {key: value for key, value in run.items() if key != 'rows'}
        if separate_low_effort:
            expected = copy.deepcopy(baseline['protocol'])
            expected['chat_template_kwargs']['reasoning_effort'] = 'low'
            if run['protocol'] != expected:
                raise ValueError('Separate effort audit permits only the declared low-effort change')
            report['gsm8k_comparison'] = None
            report['auxiliary_comparison'] = None
        else:
            report['gsm8k_comparison'] = compare(baseline_dir, metrics[0].parent)
        command = [sys.executable, str(ROOT / 'bench/rescore_quality.py'),
                   '--run-dir', str(metrics[0].parent), '--output', str(output / 'official-rescore.json')]
        report['official_rescore_command'] = command
        write_json(destination, report)
        with (output / 'official-rescore.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        report['official_rescore_returncode'] = result.returncode
        if result.returncode:
            raise ValueError('Official rescoring failed; inspect the retained log/report')
        rescore = json.loads((output / 'official-rescore.json').read_text())
        validate_rescore(rescore, run)
        report['official_rescore_sha256'] = hashlib.sha256((output / 'official-rescore.json').read_bytes()).hexdigest()
        if not separate_low_effort:
            report['auxiliary_comparison'] = compare_battery(
                ROOT / 'results/quality/baseline-logprob-chunk128/battery.json',
                ROOT / f'results/quality/{tag}/battery.json')
        for data in (baseline, run):
            for name, digest in data['evidence_sha256'].items():
                if hashlib.sha256((Path(data['directory']) / name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Quality evidence changed during audit')
        report.update(status='completed_audit')
    except Exception as error:
        report.update(status='failed_evidence', error=f'{type(error).__name__}: {error}')
    finally:
        write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--separate-low-effort', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing evidence; output must be fresh')
    result = audit(args.tag, args.output.resolve(), args.separate_low_effort)
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] == 'completed_audit' else 1


if __name__ == '__main__':
    raise SystemExit(main())
