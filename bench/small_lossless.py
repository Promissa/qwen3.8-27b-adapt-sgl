#!/usr/bin/env python3
"""Bounded diagnostic: two failing prompts, two held-out prompts, then short repeats."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from common import ROOT, api, now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--base-url', default='http://127.0.0.1:30000')
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve previous evidence; use a fresh output')
    corpus = (ROOT / 'bench/prompts_real.jsonl').read_bytes()
    prompts = [json.loads(x)['prompt'] for x in corpus.decode().splitlines()]
    cases = [(6,160,'debug'), (7,160,'debug'), (0,320,'held-out'), (3,320,'held-out')]
    result = {'at':now(), 'status':'running', 'protocol':{'cases':cases,
        'corpus_sha256':hashlib.sha256(corpus).hexdigest(), 'temperature':0,'top_k':1},
        'server_info':api(args.base_url, '/get_server_info'), 'outputs':[], 'bench':[],
        'scope':'Bounded strict greedy check, then six repeated real requests. No full-quality or universal losslessness claim.'}
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and (reference['status']!='passed' or reference['protocol']!=json.loads(json.dumps(result['protocol']))):
        raise ValueError('Require a completed matched control')
    write_json(args.output,result)
    for idx,(prompt_index,cap,phase) in enumerate(cases):
        api(args.base_url,'/flush_cache',{})
        t=time.perf_counter()
        response=api(args.base_url,'/generate',{'text':prompts[prompt_index],
            'sampling_params':{'temperature':0,'top_k':1,'max_new_tokens':cap},'return_logprob':True})
        elapsed=time.perf_counter()-t
        ids=response.get('output_ids')
        if ids is None:ids=[x[1] for x in response['meta_info']['output_token_logprobs']]
        if not ids or len(ids)!=response['meta_info']['completion_tokens']:
            raise ValueError('Incomplete token capture')
        row={'prompt_index':prompt_index,'cap':cap,'phase':phase,'output_ids':ids,
            'elapsed_s':elapsed,'meta_info':response['meta_info']}
        if reference:
            ref=reference['outputs'][idx]['output_ids']; row['identical']=ref==ids
            row['first_difference']=next((i for i,(a,b) in enumerate(zip(ref,ids)) if a!=b),
                min(len(ref),len(ids)) if len(ref)!=len(ids) else None)
        if not reference:
            retained=json.loads((ROOT/'results/experiments/deterministic-control-triton-eager-r1/tokens.json').read_text())
            row['matches_retained_control_prefix']=ids==retained['outputs'][prompt_index]['output_ids'][:cap]
            if not row['matches_retained_control_prefix']:
                result['outputs'].append(row);result['status']='control_changed';write_json(args.output,result);return 1
        result['outputs'].append(row);write_json(args.output,result)
        if reference and not row['identical']:
            result['status']='strict_failed';write_json(args.output,result);return 1
    # Repeated requests have identical sampling on both boots and flush only the
    # radix cache. Store token IDs too: a timing result cannot waive equality.
    for repeat in range(3):
        for prompt_index in (0,3):
            api(args.base_url,'/flush_cache',{})
            t=time.perf_counter();response=api(args.base_url,'/generate',{'text':prompts[prompt_index],
                'sampling_params':{'temperature':0,'top_k':1,'max_new_tokens':256,
                                   'min_new_tokens':256,'ignore_eos':True},'return_logprob':True})
            elapsed=time.perf_counter()-t;ids=response.get('output_ids')
            if ids is None:ids=[x[1] for x in response['meta_info']['output_token_logprobs']]
            if len(ids)!=256 or response['meta_info']['completion_tokens']!=256:
                raise ValueError('Incomplete fixed-length benchmark')
            row={'repeat':repeat,'prompt_index':prompt_index,'output_ids':ids,
                 'elapsed_s':elapsed,'output_tok_s':len(ids)/elapsed,'meta_info':response['meta_info']}
            if reference:
                control=reference['bench'][len(result['bench'])];row['identical']=ids==control['output_ids']
                if not row['identical']:
                    result.update(status='strict_benchmark_failed');result['bench'].append(row);write_json(args.output,result);return 1
            result['bench'].append(row);write_json(args.output,result)
    result['status']='passed';result['finished_at']=now();write_json(args.output,result);return 0

if __name__=='__main__':raise SystemExit(main())
