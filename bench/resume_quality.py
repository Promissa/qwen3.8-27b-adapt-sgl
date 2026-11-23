#!/usr/bin/env python3
"""Complete interrupted GSM8K with retained responses and official rescoring."""
import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from common import ROOT, api, now, write_json
from eval_quality import gsm8k_preflight


def validate_serving_configuration(original, current, request_seed):
    # Server restart timings/state and its default RNG seed can change. A
    # fixed per-request seed is required before ignoring that default seed.
    ignored = {'internal_states', 'startup_time', 'random_seed'}
    required = {'model_path', 'kv_cache_dtype', 'mamba_ssm_dtype', 'attention_backend',
                'prefill_attention_backend', 'decode_attention_backend', 'cuda_graph_config',
                'enable_deterministic_inference', 'speculative_algorithm'}
    if type(request_seed) is not int:
        raise ValueError('Configuration identity requires an integer per-request seed')
    if not all(isinstance(info, dict) and required <= info.keys() for info in (original, current)):
        raise ValueError('Missing original/current serving configuration evidence')
    before = {k: v for k, v in original.items() if k not in ignored}
    after = {k: v for k, v in current.items() if k not in ignored}
    if before != after:
        changed = sorted(k for k in before.keys() | after.keys()
                         if k not in before or k not in after or before[k] != after[k])
        raise ValueError('Resumed serving configuration differs: ' + ', '.join(changed))
    return {'compared_fields': len(before), 'excluded_restart_fields': sorted(ignored),
            'per_request_seed': request_seed,
            'configuration_sha256': hashlib.sha256(json.dumps(before, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()}


def validate_experiment_environment(original, current):
    if not isinstance(original, dict):
        raise ValueError('Missing original experimental environment evidence')
    select = lambda env: {k: v for k, v in env.items() if k.startswith('SGLANG_') or k == 'MAX_JOBS'}
    before, after = select(original), select(current)
    if before != after:
        changed = sorted(k for k in before.keys() | after.keys()
                         if k not in before or k not in after or before[k] != after[k])
        raise ValueError('Resumed experimental environment differs: ' + ', '.join(changed))
    return {'compared_fields': len(before),
            'environment_sha256': hashlib.sha256(json.dumps(before, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()}


def validate_rows(rows, examples, cap):
    known = {example.id: example for example in examples}
    retained = {}
    for row in rows:
        identifier = row['id']
        if identifier not in known or identifier in retained:
            raise ValueError('Unknown or duplicate retained example: ' + identifier)
        example = known[identifier]
        if (row['problem'], row['expected_answer']) != (example.inputs['problem'], str(example.target)):
            raise ValueError('Retained question or gold differs: ' + identifier)
        tokens = row['num_generated_tokens']
        if row['finish_reason'] not in ('stop', 'length') or not isinstance(tokens, int) or not 0 < tokens <= cap:
            raise ValueError('Unsuccessful or invalid retained generation: ' + identifier)
        if row['finish_reason'] == 'length' and tokens != cap:
            raise ValueError('Retained output cap differs: ' + identifier)
        if not isinstance(row.get('generation'), str) or not isinstance(row.get('symbolic_correct'), bool):
            raise ValueError('Incomplete retained generation: ' + identifier)
        retained[identifier] = row
    return retained


def source_protocol(probe_directory):
    prefix = 'sgl-eval gsm8k generation: '
    matches = []
    for path in probe_directory.glob('validation-*.log'):
        with path.open(errors='replace') as stream:
            first = stream.readline(4096)
        if first.startswith(prefix):
            matches.append(ast.literal_eval(first[len(prefix):].strip()))
    if len(matches) != 1:
        raise ValueError('Require one recorded original generation protocol')
    protocol, = matches
    expected = {'max_tokens', 'temperature', 'top_p', 'min_p', 'seed', 'chat_template_kwargs', 'top_k'}
    if set(protocol) != expected or protocol['seed'] is None:
        raise ValueError('Unsupported or unseeded resume protocol')
    return protocol


def sample_function(retained, fresh, prompt_lengths):
    from sgl_eval.types import Sample

    def sample(example, repeat):
        if repeat != 0:
            raise ValueError('Resume supports the one-repeat quality protocol')
        row = retained.get(example.id)
        if row is None:
            return fresh(example, repeat)
        return Sample(text=row['generation'], reasoning_content=row.get('reasoning_content'),
            completion_tokens=row['num_generated_tokens'],
            reasoning_tokens=row.get('num_reasoning_tokens'),
            prompt_tokens=prompt_lengths[int(example.id.removeprefix('gsm8k-'))],
            finish_reason=row['finish_reason'], generation_start_time=row.get('generation_start_time'),
            generation_end_time=row.get('generation_end_time'))

    return sample


def main():
    import yaml
    import sgl_eval.evals._loader as loader
    from sgl_eval.evals._math import MathEvaluator, make_sample_fn, make_score_one_fn
    from sgl_eval.evals._prompts import resolve_prompt
    from sgl_eval.predictions import PredictionsWriter
    from sgl_eval.registry import get
    from sgl_eval.runner import run_examples
    from sgl_eval.sampler import ChatCompletionSampler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--source-probe', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:30000')
    args = parser.parse_args()
    source = args.source_run.resolve()
    output = ROOT / 'results/gsm8k' / args.tag
    if output.exists():
        parser.error('Use a fresh resumed evaluation tag')
    original = json.loads(args.source_probe.read_text())
    protocol = source_protocol(args.source_probe.parent)
    prior_preflight = json.loads((source.parent / 'preflight.json').read_text())
    info = api(args.base_url, '/get_server_info')
    serving_identity = validate_serving_configuration(original.get('server_info'), info, protocol['seed'])
    environment_identity = validate_experiment_environment(original.get('experiment_environment'), os.environ)
    command = original['command']
    model_indices = [i for i, value in enumerate(command) if value == '--model-path']
    previous_model = Path(command[model_indices[-1] + 1]).resolve()
    if Path(info['model_path']).resolve() != previous_model:
        raise ValueError('Resumed checkpoint path differs')
    if json.loads((previous_model / 'revision.json').read_text()) != original['target_revision']:
        raise ValueError('Checkpoint revision changed')
    provenance = original['runtime_provenance']
    engine = provenance['engine_source']
    revision = subprocess.check_output(['git', '-C', engine, 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', engine, 'diff', '--binary', 'HEAD'])
    if revision != provenance['engine_commit'] or hashlib.sha256(diff).hexdigest() != provenance['engine_diff_sha256']:
        raise ValueError('Serving source changed since the retained generation')
    for package, version in provenance['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Runtime package changed: ' + package)
    effort = protocol['chat_template_kwargs'].get('reasoning_effort')
    gsm8k_preflight(args.base_url, protocol['max_tokens'], effort, output / 'preflight.json')
    current = json.loads((output / 'preflight.json').read_text())
    for key in ('n', 'chat_template_kwargs', 'max_output_tokens', 'prompt_lengths',
                'prompt_template_sha256', 'questions_sha256'):
        if current[key] != prior_preflight[key]:
            raise ValueError('Resumed prompt protocol changed: ' + key)
    loader._CACHE_ROOT = ROOT / '.cache/sgl_eval'
    examples = loader.load_via_prepare('gsm8k', ('test',))(1319)
    if len(examples) != 1319:
        raise ValueError('Require all1319 official examples')
    prediction_file = source / 'output-rs0.jsonl'
    saved_bytes = prediction_file.read_bytes()
    rows = [json.loads(line) for line in saved_bytes.splitlines() if line.strip()]
    retained = validate_rows(rows, examples, protocol['max_tokens'])
    spec = get('gsm8k')
    gen = replace(spec.default_gen, **{k: protocol[k] for k in
        ('max_tokens', 'temperature', 'top_p', 'min_p', 'seed', 'chat_template_kwargs')},
        extra_body={'top_k': protocol['top_k']})
    prompt = resolve_prompt('math')
    prompt_config = yaml.safe_load(prompt.read_text())
    if prompt_config.get('system') is not None:
        gen = replace(gen, system_message=prompt_config['system'].format())
    sampler = ChatCompletionSampler(base_url=args.base_url.rstrip('/') + '/v1', model='qwen3.8-27b')
    run_directory = output / 'sgl_eval_gsm8k_resumed'
    metadata = {'at': now(), 'status': 'running', 'source_run': str(source),
        'resume_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'serving_configuration_identity': serving_identity,
        'experimental_environment_identity': environment_identity,
        'source_prediction_sha256': hashlib.sha256(saved_bytes).hexdigest(),
        'source_probe': str(args.source_probe), 'source_probe_sha256': hashlib.sha256(args.source_probe.read_bytes()).hexdigest(),
        'generation': protocol, 'retained_ids': sorted(retained), 'retained_count': len(retained),
        'planned_new_count': 1319 - len(retained),
        'scope': 'Official runner and MathEvaluator regrade all1319 examples. Retained responses preserve their original token counts and timestamps. New generation requests cover the missing IDs.'}
    write_json(output / 'resume.json', metadata)
    print('sgl-eval gsm8k generation: ' + str(protocol), flush=True)
    try:
        with PredictionsWriter(run_directory, 1, spec.pred_schema) as writer:
            result = run_examples(name='gsm8k', examples=examples,
                sample_fn=sample_function(retained, make_sample_fn(sampler, gen, prompt), current['prompt_lengths']),
                score_one_fn=make_score_one_fn(MathEvaluator(config=prompt_config.get('evaluator_config', {}))),
                num_threads=1, n_repeats=1, on_sample_scored=writer)
        if result.partial or result.num_examples != 1319 or result.aggregate.get('error_rate', 0) != 0:
            raise RuntimeError('Resumed evaluation is incomplete or contains request errors')
        complete = [json.loads(line) for line in (run_directory / 'output-rs0.jsonl').read_text().splitlines()]
        by_id = {row['id']: row for row in complete}
        if len(complete) != 1319 or len(by_id) != 1319:
            raise RuntimeError('Resumed prediction coverage differs from1319')
        for identifier, old in retained.items():
            row = by_id[identifier]
            if any(row[key] != old[key] for key in ('generation', 'finish_reason', 'num_generated_tokens',
                                                    'symbolic_correct', 'predicted_answer')):
                raise RuntimeError('Retained response or official score changed: ' + identifier)
        write_json(run_directory / 'metrics.json', {'name': 'gsm8k', 'model': sampler.model,
            'generation': protocol, 'aggregate': result.aggregate, 'num_examples': result.num_examples,
            'n_repeats': 1, 'latency_seconds': None, 'output_throughput_tps': None,
            'resume_metadata': str(output / 'resume.json')})
        metadata.update(status='completed', new_count=1319 - len(retained),
                        resumed_segment_wall_seconds=result.latency)
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sampler.abort()
        write_json(output / 'resume.json', metadata)


if __name__ == '__main__':
    main()
