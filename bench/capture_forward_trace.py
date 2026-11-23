#!/usr/bin/env python3
"""Arm exactly one short real prompt after the owned probe's capacity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from common import ROOT, api, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:30000')
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--arm-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.arm_file.exists():
        parser.error('Require fresh output and arm file')
    manifests = list(args.trace_root.glob('worker-*/trace.json'))
    if len(manifests) != 1:
        parser.error('Require exactly one bootstrapped target worker; disable both graph phases')
    initial = json.loads(manifests[0].read_text())
    if initial['status'] != 'unarmed' or initial['frames']:
        parser.error('Existing trace must be unarmed with no capacity/warmup tensor dumps')
    collector = ROOT / 'bench/forward_trace.py'
    if initial['source_sha256'] != hashlib.sha256(collector.read_bytes()).hexdigest():
        parser.error('Loaded collector source differs from current source')
    corpus = ROOT / 'bench/prompts_real.jsonl'
    prompt = json.loads(corpus.read_text().splitlines()[0])['prompt']
    tokenization = api(args.base_url, '/tokenize', {'prompt': prompt, 'add_special_tokens': True})
    prompt_ids = tokenization.get('tokens')
    if (not isinstance(prompt_ids, list) or not prompt_ids
            or any(type(token) is not int or token < 0 for token in prompt_ids)
            or tokenization.get('count') != len(prompt_ids)):
        parser.error('Require the owned server tokenizer to return one explicit prompt ID list')
    result = {'at': now(), 'status': 'running', 'prompt': prompt,
              'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
              'prompt_tokenization': tokenization,
              'sampling_params': {'temperature': 0, 'top_k': 1, 'max_new_tokens': 8},
              'server_info': api(args.base_url, '/get_server_info'),
              'trace_manifest': str(manifests[0])}
    write_json(args.output, result)
    api(args.base_url, '/flush_cache', {})
    arm_text = json.dumps({'output': str(args.output.resolve()), 'at': now()}) + '\n'
    with args.arm_file.open('x') as stream:
        stream.write(arm_text)
    try:
        response = api(args.base_url, '/generate', {'text': prompt,
            'sampling_params': result['sampling_params'], 'return_logprob': True})
        result['response'] = response
        trace = json.loads(manifests[0].read_text())
        ids = response.get('output_ids') or [row[1] for row in response['meta_info']['output_token_logprobs']]
        if len(ids) != 8 or response['meta_info']['completion_tokens'] != 8:
            raise RuntimeError('Require all eight actual generated tokens')
        if trace['status'] != 'complete' or len(trace['frames']) != 2:
            raise RuntimeError('Missing complete actual prefill/first-step trace')
        if trace['frames'][1]['input_ids'][0] != ids[0]:
            raise RuntimeError('First step root does not match the first emitted token')
        prefill = trace['frames'][0]
        if prefill['input_ids'] != prompt_ids:
            raise RuntimeError('Captured prefill does not match the requested prompt token IDs')
        if prefill['positions'] != list(range(len(prefill['input_ids']))) or trace['frames'][1]['positions'][0] != len(prefill['input_ids']):
            raise RuntimeError('Trace did not capture one complete flushed prefill and its immediate next position')
        result.update(status='passed', output_ids=ids,
            scope='Eight-token real prompt and two actual forward snapshots; not strict512-token gate, full quality or timing')
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if args.arm_file.read_text() != arm_text:
            result.update(status='failed', cleanup_error='Arm file changed; retained for inspection')
        else:
            args.arm_file.unlink()
        write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
