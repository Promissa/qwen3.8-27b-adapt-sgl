#!/usr/bin/env python3
"""Serial continuation of the revised memory/decode/TTFT targets, preserving every trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"bench"))
from common import now,write_json
from mini_probe import idle
PY=str(ROOT/".venv/bin/python")

def environment(model):
    env=dict(os.environ,MODEL_PATH=str(ROOT/"weights"/model),MAX_JOBS="2")
    for key in list(env):
        if key.startswith("SGLANG_OPT_") or key=="DRAFT_MODEL_PATH":
            env.pop(key,None)
    env.update(SGLANG_ENABLE_LOGPROB_CHUNK="1",SGLANG_LOGPROB_CHUNK_SIZE="128")
    return env

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase",choices=["memory","spec","serving"],required=True)
    args=p.parse_args()
    if args.output.exists():p.error("Use a fresh run directory")
    idle(30000);args.output.mkdir(parents=True)
    state={"started_at":now(),"status":"running","phase":args.phase,"trials":[],
           "targets":{"fixed_memory_saving_gib":1.5,"balanced_input_tokens":48640,"max_input_tokens":81400,
              "decode_tok_s":[90,100],"cold_ttft_ms":[500,550],"warm_ttft_ms":50},
           "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ["scripts/continue_targets.py","bench/target_metrics.py","bench/mini_probe.py"]}}
    write_json(args.output/"runs.json",state)
    cfgs={c["name"]:c for c in json.loads((ROOT/"configs/experiments.json").read_text())["experiments"]}
    def trial(tag,model,profile,flags,steps,extra_env=None):
        idle(30000)
        env=environment(model);env.update(extra_env or {})
        plan=args.output/f"{tag}-plan.json";write_json(plan,steps)
        cmd=[PY,"bench/mini_probe.py","--profile",profile,"--output",str(args.output/tag/"probe"),
             "--validation-plan",str(plan),"--",*flags]
        with (args.output/f"{tag}.log").open("x") as log:
            run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        row={"tag":tag,"model":model,"returncode":run.returncode,"finished_at":now(),
             "probe":str(args.output/tag/"probe/probe.json")}
        state["trials"].append(row);write_json(args.output/"runs.json",state)
        idle(30000)
        return run.returncode==0
    try:
        if args.phase=="memory":
            # Keep KV capacity, state slots, backend, graphs and metadata format matched.
            flags=["--max-total-tokens","4096","--max-mamba-cache-size","4","--chunked-prefill-size","1024"]
            for index,model in enumerate(["target-mixed-control","target-embedding-nvfp4",
                                          "target-embedding-nvfp4","target-mixed-control"]):
                tag=f"memory-{index}-{model}"
                steps=[{"command":[PY,"bench/target_metrics.py","--output",str(args.output/tag/"metrics.json"),
                         "--lengths","1024","--output-tokens","128","--repeats","1"],"timeout_s":180}]
                if not trial(tag,model,"balanced",flags,steps):
                    raise RuntimeError("Memory trial failed; preserve error before changing conditions")
        elif args.phase=="spec":
            control=args.output/"strict-control.json"
            for role,name in [("strict-control","deterministic-control-triton-eager"),
                              ("strict-ngram-decode","deterministic-ngram-triton-eager")]:
                cfg=cfgs[name]
                cmd=[PY,"bench/token_compare.py","--output",str(args.output/f"{role}.json")]
                if role!="strict-control":cmd.extend(["--reference",str(control)])
                steps=[{"command":cmd,"timeout_s":900}]
                env=dict(cfg["env"])
                if role!="strict-control":env["SGLANG_OPT_MHA_VERIFY_DECODE"]="1"
                # Identical small pool on both boots; this is a correctness diagnostic.
                flags=[*cfg["flags"],"--max-total-tokens","4096"]
                if not trial(role,"target",cfg["profile"],flags,steps,env):
                    raise RuntimeError("Strict gate failed; no speed promotion")
        else:
            flags=["--max-total-tokens","16384","--max-mamba-cache-size","4","--chunked-prefill-size","2048"]
            for index,model in enumerate(["target-mixed-control","target-embedding-nvfp4"]):
                tag=f"serving-{index}-{model}"
                steps=[{"command":[PY,"bench/target_metrics.py","--output",str(args.output/tag/"metrics.json"),
                                   "--repeats","2"],
                        "timeout_s":900}]
                if not trial(tag,model,"balanced",flags,steps):
                    raise RuntimeError("Matched serving trial failed")
        state["status"]="completed"
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
    finally:
        state["finished_at"]=now();write_json(args.output/"runs.json",state)
    return 0 if state["status"]=="completed" else 1
if __name__=="__main__":raise SystemExit(main())
