"""Serial auto-sized candidate, then matched reference with the same pool and inputs."""

import argparse
import json
from pathlib import Path
import subprocess

from continue_targets import ROOT, PY, environment
from common import write_json
from mini_probe import idle


def run_matrix(root):
    def run_role(folder, role, fraction=None):
        trial = folder / role
        if (trial / "exit.json").exists():
            return json.loads((trial / "exit.json").read_text())["code"]
        if trial.exists():
            raise RuntimeError(f"Trial already exists without a recorded exit: {trial}")
        cmd = [
            PY,
            str(Path(__file__).resolve()),
            "--root",
            str(folder),
            "--roles",
            role,
        ]
        if fraction is not None:
            cmd += ["--mem-fraction", str(fraction)]
        return subprocess.run(cmd, cwd=ROOT).returncode

    for role in ("mtp", "mtp-control", "mtp-fit", "mtp-fit-control"):
        if run_role(root, role):
            raise RuntimeError(f"Failed {role}; inspect its probe before retrying")
    # Keep full-capacity timeouts, then find an operating point without host changes.
    for family, fractions in [
        ("gdn", (None, 0.815, 0.79, 0.76)),
        ("max", (None, 0.8775, 0.835, 0.81, 0.78)),
    ]:
        for fraction in fractions:
            folder = (
                root
                if fraction is None
                else root / f"fraction-{family}-{str(fraction).split('.')[1]}"
            )
            if folder != root:
                folder.mkdir(exist_ok=True)
                for name in ("mtp", "natural-length-index.json"):
                    if not (folder / name).exists():
                        (folder / name).symlink_to(Path("..") / name)
            if run_role(folder, family, fraction):
                boundary = folder / family / "boundary.json"
                validation = folder / family / "probe/validation-00.log"
                if (
                    not boundary.exists()
                    or json.loads(boundary.read_text())["status"] == "passed"
                    or not validation.exists()
                    or "timed out" not in validation.read_text()
                ):
                    raise RuntimeError(f"Inspect failed trial {folder / family}")
                continue
            if family == "gdn" and run_role(folder, "gdn-control", fraction):
                raise RuntimeError(f"Failed GDN control in {folder}")
            break
        else:
            raise RuntimeError(f"No tested memory fraction completed for {family}")
    subprocess.run([PY, "bench/context80_summary.py", str(root)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=[
            "mtp",
            "mtp-control",
            "gdn",
            "gdn-control",
            "max",
            "mtp-fit",
            "mtp-fit-control",
        ],
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        help="Override the memory fraction for both arms of a capacity trial",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = root / "natural-length-index.json"
    if not index.exists():
        subprocess.run(
            [
                PY,
                "bench/longbench_near_limit.py",
                "--model-path",
                str(ROOT / "weights/target"),
                "--index",
                str(index),
            ],
            check=True,
            cwd=ROOT,
        )
    if args.roles is None:
        if args.mem_fraction is not None:
            parser.error("--mem-fraction requires explicit --roles")
        return run_matrix(root)
    original = ROOT / "results/continuation-20261119/application/balanced/plan.json"
    plans = json.loads(original.read_text())
    configs = {
        c["name"]: c
        for c in json.loads((ROOT / "configs/experiments.json").read_text())[
            "experiments"
        ]
    }
    for role in args.roles:
        idle(30000)
        family = role.split("-")[0]
        focus = role.startswith("mtp-fit")
        candidate = not role.endswith("-control")
        trial = root / role
        trial.mkdir(parents=True)
        if family == "mtp":
            cfg = configs[
                "deterministic-eagle-triton"
                if candidate
                else "deterministic-control-triton"
            ]
            flags = list(cfg["flags"])
            i = flags.index("--max-total-tokens")
            del flags[i : i + 2]
            env = environment("target")
            env.update(cfg["env"])
            if candidate:
                env["SGLANG_OPT_MHA_VERIFY_GRAPH"] = "1"
            profile = cfg["profile"]
        elif family == "max":
            profile = "max-context"
            flags = ["--mem-fraction-static", "0.92", "--max-mamba-cache-size", "1"]
            env = environment("target-embedding-nvfp4")
        else:
            profile = "balanced"
            flags = ["--max-mamba-cache-size", "4"]
            env = environment("target-embedding-nvfp4")
            if candidate:
                env.update(
                    SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND="1",
                    SGLANG_GDN_PREFILL_GRAPH_MAX_SEQS="1",
                )
        if family != "gdn" or not candidate:
            env.pop("SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND", None)
            env.pop("SGLANG_GDN_PREFILL_GRAPH_MAX_SEQS", None)
        if args.mem_fraction is not None:
            flags += ["--mem-fraction-static", str(args.mem_fraction)]
        boundary = root / family / "boundary.json"
        steps = []
        if focus:
            measured = json.loads(boundary.read_text())
            manifest = json.loads((boundary.parent / "natural80-ids.json").read_text())
            required = max(
                measured["input80"] + 1024,
                max(r["input_tokens"] for r in manifest["items"]) + 128,
            )
            cap = ((required + 16 + 1023) // 1024) * 1024
            flags += ["--max-total-tokens", str(cap)]
        elif candidate:
            steps.append(
                {
                    "command": [
                        PY,
                        "bench/context_limit.py",
                        "--output",
                        str(boundary),
                    ],
                    "timeout_s": 480,
                }
            )
        else:
            cap = json.loads(boundary.read_text())["pool"]
            flags += ["--max-total-tokens", str(cap)]
        steps.append(
            {
                "command": [
                    PY,
                    "bench/context80_cases.py",
                    "--boundary",
                    str(boundary),
                    "--output",
                    str(trial),
                    "--model-path",
                    env["MODEL_PATH"],
                    *(["--focus-80"] if focus else []),
                ],
                "timeout_s": 1800,
            }
        )
        # Reuse the frozen official splits and score full coverage once per configuration.
        for typ in () if family == "max" or focus else ("longbench", "bfcl"):
            for part in ("screen", "holdout") if typ == "longbench" else ("screen",):
                s = next(
                    s
                    for s in plans
                    if ("bench/" + typ + "_subset.py") in s["command"]
                    and s["command"][-1].endswith(
                        typ + "-" + part + (".json" if typ == "longbench" else "")
                    )
                )
                cmd = list(s["command"])
                cmd[-1] = str(trial / Path(cmd[-1]).name)
                if "--model-path" in cmd:
                    cmd[cmd.index("--model-path") + 1] = env["MODEL_PATH"]
                steps.append(
                    dict(command=cmd, timeout_s=1800, continue_on_failure=True)
                )
        write_json(trial / "plan.json", steps)
        with (trial / "launcher.log").open("x") as log:
            r = subprocess.run(
                [
                    PY,
                    "bench/mini_probe.py",
                    "--profile",
                    profile,
                    "--output",
                    str(trial / "probe"),
                    "--validation-plan",
                    str(trial / "plan.json"),
                    "--",
                    *flags,
                    "--enable-metrics",
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        write_json(trial / "exit.json", {"code": r.returncode})
        if r.returncode:
            raise SystemExit(f"{role} failed; see {trial / 'probe/probe.json'}")


if __name__ == "__main__":
    main()
