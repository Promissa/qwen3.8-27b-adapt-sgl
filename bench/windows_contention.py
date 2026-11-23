#!/usr/bin/env python3
"""Measure before/during/after an owned native Windows GPU resource consumer.

Run as a validation step of bench/probe.py, which owns the sole model server.
This is a contention/recovery observation, not a clean serving benchmark or
certification of CUDA sysmem policy, physical residency, or absent desktop.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from common import ROOT, api, gpu_sample, now, write_json
from probe import generate


def native_active(report, requested_mib):
    return (report.get('status') == 'active' and report.get('vendor_id') == 0x10DE
            and report.get('nominal_resource_bytes') == requested_mib * 1024**2
            and report.get('resource_count') == requested_mib // 64
            and report.get('completed_copy_passes', 0) >= 1
            and isinstance(report.get('windows_pid'), int) and report['windows_pid'] > 0)


def native_released(report, requested_mib):
    return (report.get('status') == 'completed' and report.get('resources_released') is True
            and report.get('vendor_id') == 0x10DE
            and report.get('nominal_resource_bytes') == requested_mib * 1024**2
            and report.get('completed_copy_passes', 0) >= 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:30000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mib', type=int, choices=range(128, 1025, 64), default=512)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Require a new output directory')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    output = args.output / 'contention.json'
    native_report = args.output / 'native.json'
    native_stop = args.output / 'stop-native'
    source = ROOT / 'scripts/windows-gpu-occupancy.ps1'
    result = {'at': now(), 'status': 'running', 'requests': [], 'telemetry_errors': [],
        'scope': 'Three sequential flushed8192/1024 observations per phase; owned native resources/copies, not physical-residency or fallback certification',
        'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'benchmark_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'server_info': api(args.base_url, '/get_server_info'), 'requested_native_mib': args.mib}
    stop_monitor = threading.Event()
    current_phase = ['starting']
    started = time.monotonic()
    process = None
    native_log = None

    def save():
        write_json(output, result)

    def read_native():
        # The Windows replacement crosses the WSL filesystem boundary. Retry a
        # brief missing/partial file before treating the consumer as inactive.
        for attempt in range(5):
            try:
                return json.loads(native_report.read_text(encoding='utf-8-sig'))
            except (FileNotFoundError, json.JSONDecodeError):
                if attempt < 4:
                    time.sleep(.05)
        return {}

    def monitor():
        try:
            with (args.output / 'telemetry.jsonl').open('a') as stream:
                while not stop_monitor.is_set():
                    row = gpu_sample()
                    row.update(phase=current_phase[0], elapsed_s=time.monotonic()-started,
                               host_meminfo=Path('/proc/meminfo').read_text())
                    if not row.get('gpus'):
                        result['telemetry_errors'].append(row.get('error', 'Missing GPU data'))
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    stop_monitor.wait(1)
        except Exception as error:
            result['telemetry_errors'].append(f'{type(error).__name__}: {error}')

    def phase(name):
        current_phase[0] = name
        for repeat in range(3):
            if name == 'during':
                if process.poll() is not None or not native_active(read_native(), args.mib):
                    raise RuntimeError('Native consumer not active during pressure phase')
            api(args.base_url, '/flush_cache', {})
            begin = time.time()
            response = generate(args.base_url, 8192, 1024, salt=1701)
            result['requests'].append(dict(phase=name, repeat=repeat,
                started_epoch_s=begin, finished_epoch_s=time.time(), **response))
            save()
            if name == 'during' and (process.poll() is not None or not native_active(read_native(), args.mib)):
                raise RuntimeError('Native consumer ended before pressure request completed')

    def release_native():
        if process is None:
            return
        if process.poll() is None:
            native_stop.write_text('Release owned resources\n')
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                # Use the Windows PID as well as the unique script/report
                # command identity; a WSL interop bridge signal is insufficient
                # evidence that the native process released GPU resources.
                pid = result.get('native_ready', read_native()).get('windows_pid')
                if not isinstance(pid, int) or pid <= 0:
                    raise RuntimeError('Cannot verify native process identity for cleanup')
                cleanup = result['native_command'][:7] + ['-StopOwnedPid', str(pid),
                    '-ReportPath', subprocess.check_output(['wslpath', '-w', str(native_report)], text=True).strip()]
                result['forced_cleanup_command'] = cleanup
                subprocess.run(cleanup, check=True, stdout=native_log, stderr=subprocess.STDOUT, timeout=15)
                process.wait(timeout=10)
                raise RuntimeError('Native fixture required forced cleanup')
        result['native_returncode'] = process.returncode
        result['native_final'] = read_native()
        if process.returncode != 0 or not native_released(result['native_final'], args.mib):
            raise RuntimeError('Native fixture did not release resources successfully')

    thread = threading.Thread(target=monitor, daemon=True)
    save()
    thread.start()
    try:
        phase('before')
        windows = lambda path: subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()
        command = ['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', windows(source),
            '-MiB', str(args.mib), '-Seconds', '600', '-ReportPath', windows(native_report),
            '-StopPath', windows(native_stop)]
        result['native_command'] = command
        native_log = (args.output / 'native.log').open('w')
        current_phase[0] = 'native_startup'
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=native_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 90
        while not native_active(read_native(), args.mib):
            if process.poll() is not None:
                raise RuntimeError('Native consumer exited before ready; inspect native.log')
            if time.monotonic() >= deadline:
                raise RuntimeError('Native consumer not ready within90s')
            time.sleep(.2)
        result['native_ready'] = read_native()
        save()
        phase('during')
        current_phase[0] = 'native_cleanup'
        release_native()
        phase('after')
        current_phase[0] = 'api_recovery'
        command = [sys.executable, str(ROOT / 'bench/api_smoke.py'), '--base-url', args.base_url,
                   '--output', str(args.output / 'api-after.json')]
        if args.image:
            command += ['--image', str(args.image.resolve())]
        result['api_recovery_returncode'] = subprocess.run(command, check=False).returncode
        if result['api_recovery_returncode']:
            raise RuntimeError('API recovery failed')
        groups = defaultdict(list)
        for row in result['requests']:
            groups[row['phase']].append(row['elapsed_s'])
        if set(groups) != {'before', 'during', 'after'} or any(len(x) != 3 for x in groups.values()):
            raise RuntimeError('Incomplete phase observations')
        result['elapsed_s_by_phase'] = {k:{'samples':v,'mean':statistics.mean(v),
            'median':statistics.median(v)} for k,v in groups.items()}
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            release_native()
        except Exception as error:
            result.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
        stop_monitor.set()
        thread.join(timeout=6)
        if thread.is_alive() or result['telemetry_errors']:
            result.update(status='failed', telemetry_valid=False)
        if native_log is not None:
            native_log.close()
        result.update(finished_at=now(), elapsed_s=time.monotonic()-started)
        save()
    print(json.dumps({'status':result['status'], 'result':str(output)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
