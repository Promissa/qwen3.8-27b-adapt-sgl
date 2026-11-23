#!/usr/bin/env python3
"""Recheck raw soak responses and export measured GPU/host/latency curves."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import copy

from common import ROOT, write_json
from soak import soak_evidence


def check_response(row):
    name, response = row['name'], row['response']
    if not math.isfinite(row['elapsed_s']) or row['elapsed_s'] < 0:
        raise ValueError('Invalid request duration')
    if name == 'long':
        if (response['input_tokens'], response['output_tokens']) != (8192,1024):
            raise ValueError('Long request length changed')
        if (response['meta_info']['prompt_tokens'],response['meta_info']['completion_tokens']) != (8192,1024):
            raise ValueError('Long request actual counts changed')
        return None
    if name == 'image_input':
        wrapper = response
        response = wrapper['response']
    choice = response['choices'][0]
    message = choice['message']
    if name == 'tool_calls':
        call = message['tool_calls'][0]
        city = json.loads(call['function']['arguments'])['city']
        if call['type'] != 'function' or call['function']['name'] != 'get_weather' or not isinstance(city,str) or not city:
            raise ValueError('Incorrect structured tool response')
        return None
    text = message.get('content') or ''
    if not text or choice['finish_reason'] == 'length':
        raise ValueError('Empty or truncated chat response')
    if name == 'image_input':
        if wrapper.get('expected_objects_verified') is not True or set(wrapper['expected_descriptions']) != {'red square','blue circle'}:
            raise ValueError('Missing vision object verification')
        words=re.findall(r'\w+',text.casefold())
        pairs=set(zip(words,words[1:]))
        if not {('red','square'),('blue','circle')} <= pairs:
            raise ValueError('Vision response does not name the actual objects')
        digest = wrapper['image_sha256']
        if not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('Missing image digest')
        return digest
    if name != 'short':
        if name not in ('reasoning_content','thinking_disabled','reasoning_effort','thinking_sampled') or '391' not in text:
            raise ValueError('Incorrect reasoning response')
        if name == 'thinking_disabled':
            if message.get('reasoning_content'):
                raise ValueError('Disabled thinking returned reasoning')
        elif not message.get('reasoning_content') or '<think>' in text:
            raise ValueError('Reasoning parser did not separate content')
    return None


def oom_events(text):
    events=[]
    for number,line in enumerate(text.splitlines(),1):
        if re.search(r'(?i)avoid CUDA OOM|may (?:cause|trigger) CUDA OOM',line):
            continue
        if re.search(r'(?i)CUDA out of memory|CUDA error:\s*out of memory|CUDA_ERROR_OUT_OF_MEMORY|\bOutOfMemoryError[:(]|\bCUDA OOM\b',line):
            events.append({'line':number,'message':line[:500]})
    return events


def snapshot(directory,archive=None,server_log=None):
    paths = [directory/'soak.json',directory/'telemetry.jsonl']
    raw = [p.read_bytes() for p in paths]
    if server_log is None:
        server_log=directory/'server.log'
        if not server_log.exists():
            server_log=ROOT/'results/probes'/directory.name/'server.log'
        if not server_log.exists():server_log=None
    log_evidence={'checked':server_log is not None,'scope':'Owned server startup through this snapshot, not a future interval',
                  'observed_oom_events':None}
    if server_log is not None:
        paths.append(server_log);raw.append(server_log.read_bytes())
        log_evidence.update(path=str(server_log),observed_oom_events=oom_events(raw[-1].decode(errors='replace')))
    no_observed_oom=bool(log_evidence['checked'] and not log_evidence['observed_oom_events'])
    result = json.loads(raw[0])
    telemetry_raw = raw[1]
    incomplete = bool(telemetry_raw and not telemetry_raw.endswith(b'\n'))
    if incomplete:
        if result['status'] != 'running':
            raise ValueError('Terminal telemetry has an incomplete line')
        telemetry_raw = telemetry_raw.rsplit(b'\n',1)[0] + b'\n' if b'\n' in telemetry_raw else b''
    samples = [json.loads(line) for line in telemetry_raw.splitlines() if line.strip()]
    if not samples:
        raise ValueError('No telemetry samples to plot')
    curves = []
    for sample in samples:
        if len(sample.get('gpus',[])) != 1:
            raise ValueError('Expected exactly one measured GPU')
        gpu=sample['gpus'][0]
        available=re.search(r'^MemAvailable:\s+(\d+) kB$',sample['host_meminfo'],re.M)
        if available is None:
            raise ValueError('Missing WSL MemAvailable')
        values=[sample['elapsed_s'],gpu['used_mib'],gpu['free_mib'],gpu['utilization_percent']]
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid measured telemetry')
        curves.append({'elapsed_s':sample['elapsed_s'],'used_gib':gpu['used_mib']/1024,
            'free_gib':gpu['free_mib']/1024,'host_available_gib':int(available[1])/1024**2,
            'utilization_percent':gpu['utilization_percent']})
    counts=Counter(row['name'] for row in result['requests'])
    required={'short','long','tool_calls','reasoning_content','thinking_disabled','reasoning_effort','thinking_sampled'}
    if result['vision_included']:
        required.add('image_input')
    errors=[]
    image_digests=[]
    for index,row in enumerate(result['requests']):
        try:
            digest=check_response(row)
            if digest:
                image_digests.append(digest)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            errors.append({'request':index,'error':str(error)})
    vision_errors=result.get('vision_quality_errors',[])
    terminal_vision=result.get('failed_request')
    retained_terminal_vision=bool(terminal_vision and terminal_vision.get('name')=='image_input' and terminal_vision.get('evidence'))
    all_vision_errors=vision_errors+([terminal_vision] if retained_terminal_vision else [])
    quality_error_validation=[]
    for index,row in enumerate(all_vision_errors):
        try:
            if row['name']!='image_input' or not isinstance(row.get('error'),str) or not row['error']:
                raise ValueError('Invalid recorded vision quality failure')
            evidence=copy.deepcopy(row['evidence'])
            if evidence.get('expected_objects_verified') is not False:
                raise ValueError('Failed vision was recorded as verified')
            digest=evidence['image_sha256']
            if not re.fullmatch('[0-9a-f]{64}',digest):
                raise ValueError('Missing failed-image digest')
            if set(evidence['expected_descriptions'])!={'red square','blue circle'}:
                raise ValueError('Different expected objects for failed image')
            choices=evidence['response']['choices']
            if len(choices)!=1 or choices[0]['finish_reason'] not in ('stop','length'):
                raise ValueError('Vision failure is not a completed API response')
            evidence['expected_objects_verified']=True
            try:
                check_response(dict(row,response=evidence))
            except ValueError as error:
                if str(error) not in ('Empty or truncated chat response','Vision response does not name the actual objects'):
                    raise
            else:
                raise ValueError('Raw caption does not confirm recorded vision failure')
            image_digests.append(digest)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            quality_error_validation.append({'vision_failure':index,'error':str(error)})
    if result.get('vision_cache_mode') == 'unique_generated_frames' and len(set(image_digests)) != len(image_digests):
        errors.append({'error':'Uncached vision frames reused image bytes'})
    terminal=result['status'] != 'running'
    evidence=soak_evidence(result,required,samples) if terminal else None
    gate=bool(terminal and result.get('full_plan_soak_gate') is True and evidence['full_plan_soak_gate']
              and not errors and not all_vision_errors and not quality_error_validation and no_observed_oom)
    runtime_gate=bool(terminal and result.get('runtime_stability_7200s_gate') is True
        and evidence['runtime_stability_7200s_gate'] and not errors and not quality_error_validation and no_observed_oom)
    summary={'scope':'Raw response/telemetry snapshot; running data never establishes a completed soak',
        'recorded_status':result['status'],'recorded_elapsed_s':result['elapsed_s'],
        'recorded_error':result.get('error'), 'failed_request':result.get('failed_request'),
        'requested_seconds':result['requested_seconds'],'telemetry_last_s':curves[-1]['elapsed_s'],
        'sample_count':len(samples),'incomplete_live_line_omitted':incomplete,
        'completed_case_counts':dict(counts),'raw_response_errors':errors,
        'recorded_vision_quality_error_count':len(all_vision_errors),
        'continued_vision_quality_error_count':len(vision_errors),
        'terminal_vision_failure_retained':retained_terminal_vision,
        'vision_quality_error_validation_errors':quality_error_validation,
        'vision_unique_digests':len(set(image_digests)),'recomputed_terminal_evidence':evidence,
        'full_mixed_7200s_validated':gate,
        'runtime_stability_7200s_validated':runtime_gate,
        'server_log_evidence':log_evidence,
        'raw_sources':{str(p):hashlib.sha256(data).hexdigest() for p,data in zip(paths,raw)},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for key in ('used_gib','free_gib','host_available_gib','utilization_percent'):
        values=[row[key] for row in curves]
        summary[key]={'min':min(values),'max':max(values),'mean':statistics.mean(values),
                      'first':values[0],'last':values[-1]}
    if archive is not None:
        archive.mkdir(parents=True)
        for path,data in zip(paths,raw):
            (archive/path.name).write_bytes(data)
        (archive/'report_soak.py').write_bytes(Path(__file__).read_bytes())
        summary['frozen_source_files']=[p.name for p in paths]
    return summary,curves,result,samples


def plot(summary,curves,result,samples,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    x=[row['elapsed_s']/60 for row in curves]
    for key,label in [('used_gib','GPU used'),('free_gib','GPU free')]:
        axes[0].plot(x,[row[key] for row in curves],label=label,linewidth=1)
    axes[0].set_ylabel('Device memory (GiB)');axes[0].legend(loc='upper right',ncol=2)
    axes[1].plot(x,[row['host_available_gib'] for row in curves],color='#34824c',linewidth=1)
    axes[1].set_ylabel('WSL available (GiB)')
    # Sample timestamps and monotonic elapsed are paired after the same query.
    epoch=statistics.median(datetime.fromisoformat(s['at']).timestamp()-s['elapsed_s'] for s in samples)
    groups=defaultdict(list)
    quality_rows=list(result.get('vision_quality_errors',[]))
    terminal=result.get('failed_request')
    if terminal and terminal.get('name')=='image_input' and terminal.get('evidence'):
        quality_rows.append(terminal)
    observed_requests=result['requests']+[dict(row,name='image_quality_error') for row in quality_rows]
    for row in observed_requests:
        groups[row['name']].append(((datetime.fromisoformat(row['at']).timestamp()-epoch)/60,row['elapsed_s']))
    for name,rows in groups.items():
        style={'s':35,'marker':'x','color':'#bb2525','zorder':5} if name=='image_quality_error' else {'s':10,'alpha':.7}
        axes[2].scatter([r[0] for r in rows],[r[1] for r in rows],label=name,**style)
    axes[2].set_yscale('log');axes[2].set_ylabel('Request latency (s)');axes[2].set_xlabel('Elapsed time (minutes)')
    axes[2].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    end=max([x[-1],.1]+[point[0] for rows in groups.values() for point in rows])
    for ax in axes:
        ax.grid(alpha=.2);ax.set_xlim(0,end)
    if summary['full_mixed_7200s_validated']:
        state='VALIDATED 2h mixed soak'
    elif summary['runtime_stability_7200s_validated']:
        state=f'2h runtime complete; {summary["recorded_vision_quality_error_count"]} vision quality errors; full mixed quality gate not validated'
    else:
        state='IN PROGRESS' if result['status']=='running' else 'Recorded '+result['status']+'; full mixed gate not validated'
    fig.suptitle(f'{state} | {summary["sample_count"]} samples | {len(observed_requests)} attempts',fontsize=12)
    for extension in ('png','pdf'):
        fig.savefig(output/f'curves.{extension}',dpi=160)
    plt.close(fig)
    summary['matplotlib_version']=matplotlib.__version__


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--server-log',type=Path,help='Owned server log; otherwise infer the matching probe or use a frozen local copy')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new snapshot output directory')
    summary,curves,result,samples=snapshot(args.directory,args.output,args.server_log)
    plot(summary,curves,result,samples,args.output)
    write_json(args.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('recorded_status','sample_count','completed_case_counts',
        'raw_response_errors','recorded_vision_quality_error_count','runtime_stability_7200s_validated','full_mixed_7200s_validated')}))
