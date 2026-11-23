#!/usr/bin/env python3
"""Compare caption request formats on identical image bytes and one owned server."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from api_smoke import chat, vision_check, VISION_PROMPT, contains_description
from common import api, now, write_json

OLD_PROMPT = "List the large objects as '<color> <shape>', separated by commas."

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:30000')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output file')
    data=args.image.read_bytes();sidecar=json.loads(args.image.with_suffix('.json').read_text())
    assert hashlib.sha256(data).hexdigest()==sidecar['sha256']
    expected=sidecar['expected_description']
    result={'at':now(),'status':'running','image_sha256':sidecar['sha256'],
        'expected_descriptions':expected,'old_prompt':OLD_PROMPT,'new_prompt':VISION_PROMPT,
        'protocol':'Same image bytes, temperature0,512 cap, thinking off; flush before each prompt',
        'server_info':api(args.base_url,'/get_server_info')}
    api(args.base_url,'/flush_cache',{})
    old=chat(args.base_url,[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}},
        {'type':'text','text':OLD_PROMPT}])
    result['old_response']=old
    content=old['choices'][0]['message'].get('content') or ''
    result['old_expected_objects_present']=all(contains_description(content,x) for x in expected)
    write_json(args.output,result)
    api(args.base_url,'/flush_cache',{})
    try:
        result['new_evidence']=vision_check(args.base_url,data,expected)
        result['status']='passed_new_caption_prompt'
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        if hasattr(error,'evidence'):result['new_evidence']=error.evidence
    write_json(args.output,result)
    print(json.dumps({'status':result['status'],'old_expected_objects_present':result['old_expected_objects_present']}))
    sys.exit(0 if result['status']=='passed_new_caption_prompt' else 1)
