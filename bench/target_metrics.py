#!/usr/bin/env python3
"""Matched native streaming metrics; raw SSE events retained beside the report."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request
from common import api, now, write_json

def summarize_events(events, start, finish, expected):
    counted = [(e["at"], e["data"]["meta_info"]["completion_tokens"])
               for e in events if isinstance(e["data"].get("meta_info", {}).get("completion_tokens"), int)]
    if not counted or counted[-1][1] != expected:
        raise ValueError("Missing or incomplete generation token counts")
    if any(b[1] < a[1] for a, b in zip(counted, counted[1:])):
        raise ValueError("Nonmonotonic stream counts")
    arrivals = []
    previous = 0
    for stamp, count in counted:
        if count > previous:
            arrivals.append((stamp, count))
            previous = count
    if len(arrivals) < 2:
        raise ValueError("Insufficient token arrivals for a decode measurement")
    first, last = arrivals[0], arrivals[-1]
    elapsed = finish - start
    decode_seconds = last[0] - first[0]
    if elapsed <= 0 or decode_seconds <= 0:
        raise ValueError("Invalid timing")
    return {"input_tokens": events[-1]["data"].get("meta_info", {}).get("prompt_tokens"),
            "output_tokens": expected, "ttft_ms": (first[0]-start)*1000,
            "decode_tok_s": (last[1]-first[1])/decode_seconds,
            "tpot_ms": decode_seconds*1000/(last[1]-first[1]),
            "output_throughput": expected/elapsed, "elapsed_s": elapsed,
            "first_chunk_tokens": first[1], "decode_window_tokens": last[1]-first[1],
            "final_meta": events[-1]["data"].get("meta_info", {})}

def request(base, payload, raw):
    started=time.perf_counter()
    req=urllib.request.Request(base+"/generate", data=json.dumps(dict(payload,stream=True)).encode(),
                              headers={"Content-Type":"application/json"})
    events=[]
    with raw.open("x") as out, urllib.request.urlopen(req,timeout=240) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            body=line[5:].strip()
            if body==b"[DONE]":
                break
            event={"at":time.perf_counter(),"data":json.loads(body)}
            events.append(event); out.write(json.dumps(event)+"\n")
    finish=time.perf_counter()
    result=summarize_events(events,started,finish,payload["sampling_params"]["max_new_tokens"])
    if result["input_tokens"]!=len(payload["input_ids"]):
        raise ValueError("Actual prompt length differs from requested input")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base-url",default="http://127.0.0.1:30000")
    p.add_argument("--lengths",default="1024,8192")
    p.add_argument("--output-tokens",type=int,default=1024)
    p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args()
    if args.output.exists():p.error("Preserve evidence; use a new output")
    report={"started_at":now(),"status":"running","protocol":{"input_lengths":[int(x) for x in args.lengths.split(",")],
            "output_tokens":args.output_tokens,"repeats":args.repeats,"seed":1701,"concurrency":1,
            "cache":"flush_before_every_request","return_logprob":False,"sampling":{"temperature":0,"top_k":1}},
            "server_info":api(args.base_url,"/get_server_info",timeout=10),"rows":[]}
    write_json(args.output,report)
    try:
        for rep in range(args.repeats):
            for length in report["protocol"]["input_lengths"]:
                rng=random.Random(1701+length+rep)
                ids=[rng.randrange(1000,20000) for _ in range(length)]
                api(args.base_url,"/flush_cache",{},timeout=30)
                row=request(args.base_url,{"input_ids":ids,"sampling_params":{"temperature":0,"top_k":1,
                    "max_new_tokens":args.output_tokens,"min_new_tokens":args.output_tokens,"ignore_eos":True}},
                    args.output.parent/f"{args.output.stem}-{length}-{rep}.sse.jsonl")
                row.update(repeat=rep,input_tokens=length)
                report["rows"].append(row);write_json(args.output,report)
        report["status"]="passed"
        report["summary"]={str(length):{key:statistics.median(r[key] for r in report["rows"] if r["input_tokens"]==length)
            for key in ("decode_tok_s","tpot_ms","ttft_ms","output_throughput","elapsed_s")}
            for length in report["protocol"]["input_lengths"]}
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"]=now();write_json(args.output,report)
if __name__=="__main__":main()
