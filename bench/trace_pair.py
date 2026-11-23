#!/usr/bin/env python3
"""Own serial eager control/NGRAM boots for first actual-forward localization.

Invoke only at a clean GPU boundary after reviewing the eager strict gate.
Does not interrupt, pause, or replace another controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from common import ROOT, now, write_json
from compare_forward_trace import compare


def plans(output, attention_cases=False):
    configs_path = ROOT / 'configs/experiments.json'
    configs = {c['name']: c for c in json.loads(configs_path.read_text())['experiments']}
    result = []
    for role, name in [('control', 'deterministic-control-triton-eager'),
                       ('ngram', 'deterministic-ngram-triton-eager')]:
        config = configs[name]
        directory = output / role
        trace = directory / 'tensors'
        arm = directory / 'armed'
        hooks = [{'name': 'bounded_actual_backbone', 'target_modules': ['model'],
                  'hook_factory': 'forward_trace:make_hook',
                  'config': {'output_dir': str(trace), 'arm_file': str(arm)}}]
        if attention_cases:
            hooks.append({'name': 'first_full_attention_cache', 'target_modules': ['model'],
                          'hook_factory': 'attention_case:make_hook',
                          'config': {'output_dir': str(directory / 'attention-cases'),
                                     'arm_file': str(arm), 'layer_id': 3}})
        command = [sys.executable, str(ROOT / 'bench/probe.py'), '--profile', config['profile'],
                   '--desktop', 'present', '--output', str(directory / 'probe'),
                   '--validation-plan', str(directory / 'validation-plan.json'), '--',
                   '--enable-metrics', *config['flags'], '--forward-hooks', json.dumps(hooks)]
        steps = [{'command': [sys.executable, str(ROOT / 'bench/capture_forward_trace.py'),
                  '--trace-root', str(trace), '--arm-file', str(arm),
                  '--output', str(directory / 'capture.json')], 'timeout_s': 1800}]
        result.append({'role': role, 'experiment': name, 'command': command,
                       'steps': steps, 'env': config['env']})
    return result




def source_files(attention_cases=False):
    return ['trace_pair.py', 'forward_trace.py', 'capture_forward_trace.py', 'compare_forward_trace.py'] + (
        ['attention_case.py'] if attention_cases else [])


def retained_sources(completed):
    if type(completed.get('attention_cases', False)) is not bool:
        raise ValueError('Invalid retained attention-case mode')
    names = set(source_files(completed.get('attention_cases', False)))
    if set(completed['source_sha256']) != names:
        raise ValueError('Retained collector source coverage differs')
    archived = None
    for name, digest in completed['source_sha256'].items():
        current = hashlib.sha256((ROOT / 'bench' / name).read_bytes()).hexdigest()
        if current == digest:
            continue
        if name != 'trace_pair.py':
            raise ValueError('Collector source changed after the retained trace')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid recorded runner digest')
        # The read-only verifier can be corrected independently of the capture.
        # Preserve the measured runner, and keep all three collector/comparator
        # sources plus raw tensors/engine subject to the original exact checks.
        archived = ROOT / 'results/driver-sources/archive' / ('trace-pair-' + digest + '.py')
        if not archived.is_file() or hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed verifier requires the exact archived measured runner')
    return {'measured_runner_sha256': completed['source_sha256']['trace_pair.py'],
            'verification_runner_sha256': hashlib.sha256((ROOT / 'bench/trace_pair.py').read_bytes()).hexdigest(),
            'archived_measured_runner': str(archived) if archived else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--attention-cases', action='store_true',
                        help='Also retain bounded first-full-attention cache/mask cases')
    args = parser.parse_args()
    output = args.output.resolve()
    queue = plans(output, attention_cases=True) if args.attention_cases else plans(output)
    if args.dry_run:
        print(json.dumps(queue, indent=2))
        return 0
    retained = output / 'pair.json'
    if retained.exists():
        completed = json.loads(retained.read_text())
        if (completed['status'] != 'completed_diagnostic' or completed['queue'] != queue
                or completed.get('attention_cases', False) != args.attention_cases):
            parser.error('Inspect the retained partial or different trace before retrying')
        try:
            verification_sources = retained_sources(completed)
        except ValueError as error:
            parser.error(str(error))
        for role in ('control', 'ngram'):
            probe = json.loads((output / role / 'probe/probe.json').read_text())
            provenance = probe['runtime_provenance']
            engine = provenance['engine_source']
            current = subprocess.check_output(['git', '-C', engine, 'diff', '--binary', 'HEAD'])
            revision = subprocess.check_output(['git', '-C', engine, 'rev-parse', 'HEAD'], text=True).strip()
            if revision != provenance['engine_commit'] or hashlib.sha256(current).hexdigest() != provenance['engine_diff_sha256']:
                parser.error('Serving engine changed after the retained trace')
        directories = [list((output / role / 'tensors').glob('worker-*')) for role in ('control', 'ngram')]
        if any(len(items) != 1 for items in directories):
            parser.error('Retained worker coverage differs')
        verified = compare(directories[0][0], directories[1][0])
        if json.loads(json.dumps(verified)) != json.loads((output / 'comparison.json').read_text()):
            parser.error('Retained comparison differs from its raw tensors')
        if args.attention_cases:
            from attention_case import compare_pair
            if compare_pair(output) != json.loads((output / 'attention-cache-comparison.json').read_text()):
                parser.error('Retained attention-cache comparison differs from raw cases')
        print(json.dumps({'status': 'reused_completed_diagnostic', 'manifest': str(retained),
                          **verification_sources}))
        return 0
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            argv = item.joinpath('cmdline').read_bytes().split(b'\0')
            if b'sglang.launch_server' in argv or argv[0].startswith(b'sglang::'):
                parser.error('Model process is active; inspect its owner instead of competing')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 30000))
    output.mkdir(parents=True, exist_ok=False)
    state = {'at': now(), 'status': 'running', 'queue': queue, 'trials': [],
             'attention_cases': args.attention_cases,
             'scope': 'Actual-forward localization; no timing, strict512-token or full quality promotion',
             'source_sha256': {name: hashlib.sha256((ROOT / 'bench' / name).read_bytes()).hexdigest()
                               for name in source_files(args.attention_cases)}}
    manifest = output / 'pair.json'
    write_json(manifest, state)
    try:
        configs = json.loads((ROOT / 'configs/experiments.json').read_text())['experiments']
        for job in queue:
            role = job['role']
            directory = output / role
            directory.mkdir()
            write_json(directory / 'validation-plan.json', job['steps'])
            env = dict(os.environ, MODEL_PATH=str(ROOT / 'weights/target'))
            env.pop('DRAFT_MODEL_PATH', None)
            for c in configs:
                for key in c['env']:
                    env.pop(key, None)
            env.update(job['env'])
            env['PYTHONPATH'] = str(ROOT / 'bench') + os.pathsep + env.get('PYTHONPATH', '')
            rc = subprocess.run(job['command'], env=env, cwd=ROOT).returncode
            probe = json.loads((directory / 'probe/probe.json').read_text())
            state['trials'].append({'role': role, 'returncode': rc, 'probe_status': probe['status']})
            write_json(manifest, state)
            if rc or probe['status'] != 'passed':
                raise RuntimeError('Incomplete trace probe; retained for inspection: ' + role)
        directories = [list((output / role / 'tensors').glob('worker-*')) for role in ('control', 'ngram')]
        if any(len(items) != 1 for items in directories):
            raise RuntimeError('Require exactly one actual target worker per trace')
        comparison = compare(directories[0][0], directories[1][0])
        write_json(output / 'comparison.json', comparison)
        if args.attention_cases:
            from attention_case import compare_pair
            write_json(output / 'attention-cache-comparison.json', compare_pair(output))
            state['attention_cache_comparison'] = str(output / 'attention-cache-comparison.json')
        state.update(status='completed_diagnostic', comparison=str(output / 'comparison.json'))
    except BaseException as error:
        state.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(manifest, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())
