#!/usr/bin/env python3
"""Owned serial small-scale diagnostic server; deliberately no capacity certification."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import threading
import urllib.request
from common import ROOT,api,now,write_json,gpu_sample
from probe import runtime_provenance,parse_log


def idle(port):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=p.joinpath('cmdline').read_bytes().split(b'\0')
            if b'sglang.launch_server' in a or (a and a[0].startswith(b'sglang::')):
                raise RuntimeError('Model already active; preserve its owner')
        except OSError:pass
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-plan',type=Path,required=True)
    p.add_argument('--port',type=int,default=30000)
    args,extra=p.parse_known_args()
    if extra[:1]==['--']:extra=extra[1:]
    if args.output.exists():p.error('Use a fresh diagnostic directory')
    idle(args.port);args.output.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'scripts'));from serve import command,validate_checkpoints
    cmd=command(args.profile,extra+['--port',str(args.port),'--max-running-requests','1'])
    validate_checkpoints(cmd)
    result={'started_at':now(),'status':'running_diagnostic','command':cmd,
            'runtime_provenance':runtime_provenance(args.output),'capacity_gate_passed':False,
            'scope':'Small-scale correctness/performance diagnostic only; full capacity gate not run',
            'experiment_environment':{k:v for k,v in os.environ.items() if k.startswith('SGLANG_OPT_')},
            'validation_steps':[], 'prelaunch_gpu':gpu_sample()}
    path=args.output/'probe.json';write_json(path,result);process=None
    stop=threading.Event()
    def monitor():
        with (args.output/'gpu-telemetry.jsonl').open('x') as f:
            while not stop.is_set():
                f.write(json.dumps(gpu_sample())+'\n');f.flush();stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        with (args.output/'server.log').open('w') as log:
            t=time.monotonic();process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result['server_pid']=process.pid
            result['server_start_ticks']=Path(f'/proc/{process.pid}/stat').read_text().split()[21]
            base=f'http://127.0.0.1:{args.port}'
            while True:
                if process.poll() is not None:raise RuntimeError('Server exited before READY')
                if time.monotonic()-t>240:raise TimeoutError('Diagnostic READY timeout')
                try:
                    result['server_info']=api(base,'/get_server_info',timeout=2)
                    if os.environ.get('SGLANG_DIAG_BYPASS_HEALTH_GENERATE'):
                        raise RuntimeError('Generation health bypass cannot establish readiness')
                    with urllib.request.urlopen(base+'/health_generate',timeout=5) as response:
                        if response.status!=200:raise RuntimeError('Generation readiness rejected')
                    result['generation_readiness_passed']=True
                    break
                except Exception:time.sleep(.5)
            result['ready_s']=time.monotonic()-t;write_json(path,result)
            failed_steps = []
            for step in json.loads(args.validation_plan.read_text()):
                with (args.output/f"validation-{len(result['validation_steps']):02d}.log").open('w') as f:
                    rc=subprocess.run(step['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=step.get('timeout_s',600)).returncode
                result['validation_steps'].append({'command':step['command'],'returncode':rc});write_json(path,result)
                if rc:
                    if not step.get('continue_on_failure'):
                        raise RuntimeError(f'Diagnostic step failed: {rc}')
                    failed_steps.append(len(result['validation_steps']) - 1)
            if failed_steps:
                raise RuntimeError(f'Diagnostic steps failed: {failed_steps}')
            if process.poll() is not None:raise RuntimeError('Server exited during diagnostic')
            result['status']='passed_diagnostic'
    except BaseException as e:
        result.update(status='failed_diagnostic',error=f'{type(e).__name__}: {e}')
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            result['server_returncode']=process.returncode
        stop.set();thread.join(timeout=8)
        result['server_log_evidence']=parse_log((args.output/'server.log').read_text()) if (args.output/'server.log').exists() else {}
        result['post_shutdown_gpu']=gpu_sample()
        result['finished_at']=now();write_json(path,result)
    idle(args.port)
    print(json.dumps({'status':result['status'],'error':result.get('error'),'probe':str(path)}))
    return 0 if result['status']=='passed_diagnostic' else 1

if __name__=='__main__':raise SystemExit(main())
