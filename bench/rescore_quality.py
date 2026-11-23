#!/usr/bin/env python3
"""Re-extract and officially regrade a completed GSM8K run without generation."""
import argparse
import asyncio
import importlib.metadata
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import yaml
import sgl_eval.evals._loader as loader
from sgl_eval.evals._math import MathEvaluator
from sgl_eval.evals._prompts import resolve_prompt

root = Path.cwd()
sys.path.insert(0, str(root / 'bench'))
from common import now, write_json
from compare_quality import read_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
destination = args.output.resolve()
if destination.exists():
    parser.error('Preserve existing evidence; use a fresh output file')
baseline = read_run(args.run_dir)
loader._CACHE_ROOT = root / '.cache/sgl_eval'
examples = loader.load_via_prepare('gsm8k', ('test',))(1319)
assert len(examples) == 1319 and {example.id for example in examples} == set(baseline['rows'])
prompt = resolve_prompt('math')
config = yaml.safe_load(prompt.read_text()).get('evaluator_config', {})
evaluator = MathEvaluator(config=config or {})
assert evaluator.eval_config.use_predicted_answer_key is False
result = {'at': now(), 'status': 'running', 'checked': 0, 'correct': 0, 'n': 1319,
    'score_mismatch_ids': [], 'extraction_mismatch_ids': [], 'rows': [],
    'run_directory': baseline['directory'],
    'sgl_eval_version': importlib.metadata.version('sgl-eval'),
    'original_evidence_sha256': baseline['evidence_sha256'], 'evaluator_config': config,
    'prompt_yaml_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
    'scorer_source_sha256': hashlib.sha256(Path(inspect.getsourcefile(MathEvaluator)).read_bytes()).hexdigest(),
    'verification_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'questions_and_golds_sha256': hashlib.sha256(json.dumps([(example.id, example.inputs['problem'], str(example.target)) for example in examples], sort_keys=True).encode()).hexdigest(),
    'scope': 'All1319 actual cached upstream questions/golds checked; saved generation text re-extracted and regraded by pinned official MathEvaluator. Stored predicted_answer/symbolic_correct are never passed to the scorer. No new model generation or aggregate replacement.'}
start = time.monotonic()
try:
    for example in examples:
        old = baseline['rows'][example.id]
        assert old['problem'] == example.inputs['problem'] and old['expected_answer'] == str(example.target), example.id
        scored = asyncio.run(evaluator.eval_single({'generation': old['generation'], 'expected_answer': str(example.target)}))
        correct = bool(scored['symbolic_correct'])
        if correct != old['symbolic_correct']:
            result['score_mismatch_ids'].append(example.id)
        if scored['predicted_answer'] != old['predicted_answer']:
            result['extraction_mismatch_ids'].append(example.id)
        result['rows'].append({'id': example.id, 'expected_answer': str(example.target),
            'predicted_answer': scored['predicted_answer'], 'symbolic_correct': correct})
        result['checked'] += 1
        result['correct'] += correct
        if result['checked'] % 100 == 0:
            write_json(destination, result)
            print(json.dumps({'checked': result['checked'], 'correct': result['correct']}), flush=True)
    for filename, digest in baseline['evidence_sha256'].items():
        if hashlib.sha256((Path(baseline['directory']) / filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Original evidence changed during rescoring: ' + filename)
    result.update(status='passed' if not result['score_mismatch_ids'] and not result['extraction_mismatch_ids'] else 'failed_disagreement',
        accuracy_percent=100 * result['correct'] / 1319, elapsed_s=time.monotonic() - start)
except BaseException as error:
    result.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    write_json(destination, result)
print(json.dumps({key: result[key] for key in ('status', 'checked', 'correct', 'accuracy_percent', 'score_mismatch_ids', 'extraction_mismatch_ids', 'elapsed_s')}))
raise SystemExit(0 if result['status'] == 'passed' else 1)
