#!/usr/bin/env python3
"""Serial, deadline-bounded validation requested by the user on2026-11-12."""
import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/budget-3h-20261112'
PY = str(ROOT / '.venv/bin/python')
GPU_DEADLINE = datetime.datetime.fromisoformat('2026-11-12T11:35:11+00:00')
REPORT = OUT / 'runs.json'


def remaining():
    return (GPU_DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()


def idle():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            if b'sglang.launch_server' in argv or argv[0].startswith(b'sglang::'):
                raise RuntimeError('Model still active; preserve serial GPU ownership')
        except (OSError, IndexError):
            pass
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 30000))


def save(state):
    temporary = REPORT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(REPORT)


def environment(model):
    env = dict(os.environ, MODEL_PATH=str(ROOT / 'weights' / model), MAX_JOBS='2')
    env.pop('DRAFT_MODEL_PATH', None)
    configs = json.loads((ROOT / 'configs/experiments.json').read_text())['experiments']
    for config in configs:
        for name in config['env']:
            env.pop(name, None)
    env.pop('SGLANG_OPT_MHA_VERIFY_TILE32', None)
    env.update(SGLANG_ENABLE_LOGPROB_CHUNK='1', SGLANG_LOGPROB_CHUNK_SIZE='128')
    return env


def steps(tag, model, *, quality=False, final=False, effort=False, harness=False):
    validation = f'results/validation/{tag}'
    result = [
        {'command': [PY, 'bench/api_smoke.py', '--output', f'{validation}/api.json'], 'timeout_s': 180},
        {'command': [PY, 'bench/token_compare.py', '--output', f'{validation}/tokens.json'], 'timeout_s': 240},
        {'command': [PY, 'bench/budget_screen.py', 'bench', tag], 'timeout_s': 540},
    ]
    if final:
        result.append({'command': [PY, 'bench/long_context.py', '--near-limit', '--model-path',
            str(ROOT / 'weights' / model), '--output', f'{validation}/retrieval.json'], 'timeout_s': 480})
    if quality:
        command = [PY, 'bench/eval_quality.py', '--tag', tag, '--num-examples', '64', '--max-tokens', '16384']
        audit = [PY, 'bench/budget_screen.py', 'quality-report', tag]
        if effort:
            command += ['--reasoning-effort', 'low']
            audit += ['--separate-effort']
        result.extend([
            {'command': command, 'timeout_s': 900},
            {'command': audit, 'timeout_s': 180},
            {'command': [PY, 'bench/quality_battery.py', '--windows-per-domain', '6',
                '--gsm8k-examples', '32', '--output', f'results/quality/{tag}/battery.json'], 'timeout_s': 600},
            {'command': [PY, 'bench/budget_screen.py', 'aux-report', tag], 'timeout_s': 120},
        ])
    if final:
        result.append({'command': [PY, 'bench/soak.py', '--seconds', '300', '--output',
            f'results/soak/{tag}'], 'timeout_s': 450})
    if harness:
        result.append({'command': [PY, 'bench/run_harness.py', '--seconds', '180', '--output',
            f'results/harness/{tag}'], 'timeout_s': 240})
    return result


def trial(state, tag, flags, *, model='target', profile='balanced', quality=False,
          final=False, effort=False, harness=False, maximum=1800):
    previous = [row for row in state['trials'] if row['tag'] == tag]
    if previous:
        if len(previous) != 1:
            raise ValueError('Duplicate retained budgeted trial')
        row = previous[0]
        if row['status'] in ('passed', 'failed'):
            probe = json.loads((ROOT / row['probe']).read_text())
            if probe['status'] != row['status'] or not probe.get('finished_at'):
                raise ValueError('Retained trial is not terminal')
        elif row['status'] not in ('interrupted_by_budget', 'not_run_deadline_reserve'):
            raise ValueError('Cannot adopt an incomplete retained trial')
        return
    idle()
    if remaining() < (900 if quality else 360):
        state['trials'].append({'tag': tag, 'status': 'not_run_deadline_reserve'})
        save(state)
        return
    plan = OUT / f'{tag}-plan.json'
    plan.write_text(json.dumps(steps(tag, model, quality=quality, final=final,
        effort=effort, harness=harness), indent=2) + '\n')
    command = [PY, 'bench/probe.py', '--desktop', 'present', '--profile', profile,
        '--output', f'results/probes/{tag}', '--validation-plan', str(plan), '--',
        '--enable-metrics', *flags]
    allowance = min(maximum, max(1, int(remaining() - 60)))
    state.update(current=tag, remaining_gpu_budget_s=remaining())
    save(state)
    start = datetime.datetime.now(datetime.timezone.utc)
    with (OUT / f'{tag}.log').open('w') as log:
        completed = subprocess.run(['timeout', '--signal=INT', '--kill-after=45s',
            f'{allowance}s', *command], cwd=ROOT, env=environment(model),
            stdout=log, stderr=subprocess.STDOUT)
    evidence = ROOT / f'results/probes/{tag}/probe.json'
    data = json.loads(evidence.read_text()) if evidence.exists() else {}
    row = {'tag': tag, 'model': model, 'profile': profile, 'flags': flags,
        'returncode': completed.returncode, 'status': data.get('status', 'missing_probe'),
        'budget_terminated': completed.returncode in (124, 137),
        'elapsed_s': (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds(),
        'probe': str(evidence.relative_to(ROOT)), 'reported_capacity': data.get('reported_context_capacity'),
        'error': data.get('error'), 'quality_samples_requested': 64 if quality else 0,
        'full_plan_promotion_verified': False}
    if row['budget_terminated']:
        row['status'] = 'interrupted_by_budget'
    state['trials'].append(row)
    save(state)
    idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    retained = None
    if REPORT.exists():
        if not args.resume:
            raise FileExistsError('Preserve the existing budgeted queue; explicit resume required')
        retained = json.loads(REPORT.read_text())
        if retained['status'] != 'stopped':
            raise ValueError('Resume only a reviewed stopped queue')
    state = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'running', 'controller_pid': os.getpid(),
        'controller_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'gpu_deadline': GPU_DEADLINE.isoformat(), 'trials': [],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'User-authorized reduced screening, serial GPU use; reserve30min for delivery. No1319-example/three-repeat/two-hour qualification from these short runs.'}
    if retained is not None:
        state['trials'] = retained['trials']
        state['resumed_from'] = 'results/budget-3h-20261112/runs-interrupted-r1.json'
        state['original_started_at'] = retained['started_at']
    save(state)
    try:
        pins = ['--max-mamba-cache-size', '4', '--max-total-tokens', '24576']
        trial(state, 'budget-baseline-r1', pins, maximum=600)
        trial(state, 'budget-balanced-r1', ['--max-mamba-cache-size', '4',
            '--max-total-tokens', '49152'], model='target-embedding-nvfp4',
            quality=True, final=True, harness=True, maximum=1800)
        trial(state, 'budget-max-context-r1', ['--mem-fraction-static', '0.92',
            '--max-mamba-cache-size', '1', '--max-total-tokens', '81920'],
            model='target-embedding-nvfp4', profile='max-context', quality=True, final=True, maximum=1800)
        trial(state, 'budget-qat-r1', ['--mem-fraction-static', '0.96', '--chunked-prefill-size',
            '512', *pins], model='qat', quality=True, maximum=1500)
        trial(state, 'budget-state-fp32-r1', ['--mamba-ssm-dtype', 'float32', *pins],
            quality=True, maximum=1500)
        for name, flags in [
            ('gemm-cudnn', ['--fp4-gemm-backend', 'flashinfer_cudnn']),
            ('gdn-decode-flashinfer', ['--linear-attn-decode-backend', 'flashinfer']),
            ('gdn-prefill-flashinfer', ['--linear-attn-prefill-backend', 'flashinfer']),
        ]:
            trial(state, f'budget-{name}-r1', [*pins, *flags], maximum=480)
        trial(state, 'budget-effort-low-r1', pins, quality=True, effort=True, maximum=1200)
        trial(state, 'budget-baseline-after-r1', pins, maximum=600)
        state['status'] = 'completed_budgeted_queue'
    except BaseException as error:
        state.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
