#!/usr/bin/env python3
"""Validate raw envelope probes and emit a measured table without recipe promotion."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re

from common import now, write_json
from envelope_sweep import classify


RUNNING = re.compile(r'^sglang:num_running_reqs(?:\{[^}]*\})?\s+([^\s]+)', re.MULTILINE)


def activity(probe, requested):
    telemetry = probe['concurrency_activity']['telemetry']
    active = []
    for sample in telemetry:
        values = [float(value) for value in RUNNING.findall(sample['raw'])]
        observed = max(values) if values else None
        if observed != sample['running_requests']:
            raise ValueError('Stored activity disagrees with raw exported metrics')
        if observed is not None and observed >= requested:
            active.append(sample['at'])
    span = max(active) - min(active) if active else 0.
    if len(active) < 2 or span < .5:
        raise ValueError('Concurrent capacity did not demonstrate active requests')
    return {'verified': True, 'samples': len(active), 'span_s': span,
            'metric_errors': len(probe['concurrency_activity'].get('metric_errors', []))}


def summarize(source):
    # The writer appends CSV rows and atomically replaces the search summary.
    # Retry a cross-file race; never classify a half-written row as a failure.
    for _ in range(3):
        raw_csv = source.read_bytes()
        raw_summary = source.with_suffix('.summary.json').read_bytes()
        if raw_csv == source.read_bytes():
            break
    else:
        raise RuntimeError('Envelope changed during snapshot; retry later')
    if not raw_csv.endswith(b'\n'):
        raise RuntimeError('CSV row is being written; retry later')
    rows = list(csv.DictReader(io.StringIO(raw_csv.decode())))
    desktops = {row['desktop'] for row in rows}
    if len(desktops) != 1:
        raise ValueError('Summarize one observed desktop condition at a time')
    desktop = next(iter(desktops))
    complete = {entry['combination']: entry for entry in json.loads(raw_summary)}
    points, grouped = [], {}
    for row in rows:
        evidence = Path(row['evidence'])
        raw = evidence.read_bytes()
        probe = json.loads(raw)
        concurrency, chunk = int(row['concurrency']), int(row['prefill_chunk'])
        label = f"{desktop}-{row['strategy']}-{chunk}-c{concurrency}-{row['dtype']}"
        if probe['status'] != row['status']:
            raise ValueError(f'CSV/probe status mismatch: {evidence}')
        log = '' if probe['status'] == 'passed' else evidence.with_name('server.log').read_text(errors='replace')
        if classify(probe, log) != row['outcome']:
            raise ValueError(f'CSV/probe classification mismatch: {evidence}')
        point = {'combination': label, 'fraction': float(row['mem_fraction']), 'status': probe['status'],
            'outcome': row['outcome'], 'capacity': probe.get('reported_context_capacity'),
            'peak_device_used_mib': probe.get('peak_device_used_mib'),
            'gpu_before': probe.get('gpu_before'), 'evidence': str(evidence),
            'probe_sha256': hashlib.sha256(raw).hexdigest(),
            'engine_diff_sha256': probe.get('runtime_provenance', {}).get('engine_diff_sha256')}
        if probe['status'] == 'passed':
            if not probe['capacity_gate_passed'] or int(row['capacity']) != point['capacity']:
                raise ValueError(f'Missing or inconsistent capacity proof: {evidence}')
            requests = probe['requests']
            if len(requests) != concurrency + 1:
                raise ValueError(f'Missing required capacity requests: {evidence}')
            expected = [(max(8192, int(point['capacity'] * .95) - 1), 1)] + [(8192, 1024)] * concurrency
            for request, (input_tokens, output_tokens) in zip(requests, expected):
                meta = request['meta_info']
                if (request['input_tokens'], request['output_tokens'], meta['prompt_tokens'], meta['completion_tokens']) != (input_tokens, output_tokens, input_tokens, output_tokens):
                    raise ValueError(f'Incomplete actual request token counts: {evidence}')
                if not math.isfinite(request['elapsed_s']) or request['elapsed_s'] <= 0:
                    raise ValueError(f'Invalid request time: {evidence}')
            point.update(near_limit_input_tokens=requests[0]['input_tokens'],
                near_limit_elapsed_s=requests[0]['elapsed_s'],
                fixed_request_elapsed_s=[request['elapsed_s'] for request in requests[1:]],
                fixed_request_max_elapsed_s=max(request['elapsed_s'] for request in requests[1:]),
                concurrent_activity=activity(probe, concurrency) if concurrency > 1 else None)
        grouped.setdefault(label, []).append(point)
        points.append(point)
    expected_grid = [(policy, chunk, concurrency, 'bfloat16') for policy, chunk, concurrency in
        itertools.product(('extra_buffer', 'extra_buffer_lazy', 'no_buffer', 'disabled'), (2048, 1024, 512), (1, 2))]
    expected_grid.append(('extra_buffer_lazy', 2048, 1, 'float32'))
    groups = []
    for policy, chunk, concurrency, dtype in expected_grid:
        label = f'{desktop}-{policy}-{chunk}-c{concurrency}-{dtype}'
        samples = grouped.get(label, [])
        passed = [point for point in samples if point['status'] == 'passed']
        highest = max(passed, key=lambda point: point['fraction']) if passed else None
        if label in complete:
            saved = complete[label]
            actual = {point['fraction']: point['status'] == 'passed' for point in samples}
            expected_samples = {float(key): value for key, value in saved['samples'].items()}
            if actual != expected_samples or saved['highest_tested_passing_fraction'] != (highest['fraction'] if highest else None):
                raise ValueError(f'Search summary disagrees with retained probes: {label}')
        lowest = min(passed, key=lambda point: point['fraction']) if passed else None
        groups.append({'combination': label, 'policy': policy, 'chunk': chunk, 'concurrency': concurrency, 'dtype': dtype,
            'search_status': 'completed' if label in complete else 'in_progress' if samples else 'pending',
            'sample_count': len(samples), 'passing_points': len(passed), 'highest_fraction_observation': highest,
            'fixed_elapsed_ratio_highest_vs_lowest_fraction': highest['fixed_request_max_elapsed_s'] / lowest['fixed_request_max_elapsed_s'] if highest else None,
            'global_maximum_proven': False})
    if set(grouped) - {group['combination'] for group in groups}:
        raise ValueError('Unexpected grid combination')
    return {'at': now(), 'status': 'completed_grid' if len(complete) == len(groups) else 'partial_snapshot',
        'source_csv': str(source.resolve()), 'csv_snapshot_bytes': len(raw_csv),
        'csv_snapshot_sha256': hashlib.sha256(raw_csv).hexdigest(),
        'summary_snapshot_sha256': hashlib.sha256(raw_summary).hexdigest(),
        'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'desktop': desktop, 'expected_combinations': len(groups), 'completed_combinations': len(complete),
        'point_count': len(points), 'groups': groups, 'points': points,
        'scope': 'Raw request/token/activity proof checked. Capacity-probe times are single observations, not formal repeated speed benchmarks. Desktop occupancy, fallback enforcement, full quality and final recipes remain unverified.',
        'physical_vram_maximum_proven': False, 'recipe_promotion_verified': False}


def markdown(report):
    lines = ['# Measured envelope snapshot', '',
        f"Recorded {report['at']}; {report['desktop']} desktop; {report['completed_combinations']}/{report['expected_combinations']} searches completed; {report['point_count']} points.", '',
        'Highest passing fraction is only the highest tested observation. Times below belong to capacity requests, not repeated formal benchmarks. No physical-VRAM maximum or final recipe is certified.', '',
        '| Policy | State | Chunk | C | Search | Highest pass | Pool tokens | Fixed request max s | Ratio vs lowest fraction | Probe |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---|']
    for group in report['groups']:
        point = group['highest_fraction_observation']
        values = (f"{point['fraction']:.5f}", str(point['capacity']), f"{point['fixed_request_max_elapsed_s']:.2f}",
                  f"{group['fixed_elapsed_ratio_highest_vs_lowest_fraction']:.2f}", f"[raw]({point['evidence']})") if point else ('—',) * 5
        lines.append('| ' + ' | '.join([group['policy'], group['dtype'], str(group['chunk']), str(group['concurrency']), group['search_status'], *values]) + ' |')
    lines += ['', 'C2 passes require independently reparsed raw metrics showing at least two active requests for0.5s. Near-limit input sizes vary with the measured pool. Each raw probe retains its engine diff hash and sampled device usage. Changing desktop/residency states prevent causal comparisons across these observations.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for summary.json and table.md')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new snapshot directory')
    report = summarize(args.csv)
    args.output.mkdir(parents=True)
    write_json(args.output / 'summary.json', report)
    (args.output / 'table.md').write_text(markdown(report))
    print(json.dumps({key: report[key] for key in ('status', 'point_count', 'completed_combinations', 'expected_combinations')}))
