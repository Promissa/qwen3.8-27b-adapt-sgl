#!/usr/bin/env python3
"""Locate raw greedy-token/logprob divergence without inferring a kernel cause."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from common import write_json


def analyze(reference, candidate):
    paths = [Path(reference), Path(candidate)]
    captures = [json.loads(path.read_text()) for path in paths]
    if not all(c.get('complete') for c in captures) or captures[0]['protocol'] != captures[1]['protocol']:
        raise ValueError('Require complete captures with identical prompts/sampling')
    count = captures[0]['protocol']['prompt_count']
    if count < 1 or any(len(c['outputs']) != count for c in captures):
        raise ValueError('Capture rows disagree with protocol')
    rows = []
    for index, (left, right) in enumerate(zip(*(c['outputs'] for c in captures))):
        a, b = left['output_ids'], right['output_ids']
        for ids, row in ((a, left), (b, right)):
            if not ids or any(not isinstance(t,int) or isinstance(t,bool) for t in ids):
                raise ValueError('Missing actual integer token IDs')
            if row['meta_info']['completion_tokens'] != len(ids):
                raise ValueError('Actual completion count differs from captured IDs')
        prefix = next((i for i,(x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))
        lps = []
        for row, ids in ((left,a), (right,b)):
            values = row['meta_info']['output_token_logprobs']
            if len(values) != len(ids) or [v[1] for v in values] != ids:
                raise ValueError('Logprob order differs from actual output tokens')
            if any(not isinstance(v[0],(int,float)) or not math.isfinite(v[0]) for v in values):
                raise ValueError('Missing finite chosen-token probabilities')
            lps.append([v[0] for v in values])
        shared_deltas = [abs(x-y) for x,y in zip(lps[0][:prefix],lps[1][:prefix])]
        first_lp = next((i for i,delta in enumerate(shared_deltas) if delta != 0), None)
        rows.append({'prompt_index':index, 'reference_tokens':len(a), 'candidate_tokens':len(b),
            'token_identical':a == b, 'common_prefix_tokens':prefix,
            'first_differing_position_zero_based':None if a == b else prefix,
            'reference_token_at_difference':a[prefix] if prefix < len(a) else None,
            'candidate_token_at_difference':b[prefix] if prefix < len(b) else None,
            'first_shared_token_logprob_difference_zero_based':first_lp,
            'maximum_shared_token_logprob_abs_difference':max(shared_deltas,default=0),
            'first_emitted_token_and_logprob_identical':a[0] == b[0] and lps[0][0] == lps[1][0],
            'candidate_spec_accept_length':right['meta_info'].get('spec_accept_length'),
            'candidate_spec_verify_ct':right['meta_info'].get('spec_verify_ct')})
    return {'scope':'CPU analysis of complete saved greedy outputs; chosen-token probabilities are not full logit distributions and do not establish a causal operator',
        'reference':str(paths[0].resolve()), 'candidate':str(paths[1].resolve()),
        'source_sha256':[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        'prompt_count':count, 'exact_matches':sum(row['token_identical'] for row in rows),
        'first_token_and_logprob_exact_count':sum(row['first_emitted_token_and_logprob_identical'] for row in rows),
        'common_prefix_min':min(row['common_prefix_tokens'] for row in rows),
        'common_prefix_median':statistics.median(row['common_prefix_tokens'] for row in rows),
        'rows':rows}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    write_json(args.output,analyze(args.reference,args.candidate))
